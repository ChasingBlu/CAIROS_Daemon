#include "granite_embedder.hpp"

/*
 * Granite Hidden-State CLI (C++)
 *
 * Purpose:
 *  - Diagnostics-only CLI to dump token-level model outputs (seq x dim)
 *    using the ONNX output already present in the model.
 *  - No model modification, no ONNX re-export. Reads what the model exposes.
 *
 * REPA/ISO notes:
 *  - No silent fallback: exits non-zero on errors.
 *  - Explicit backend selection (ONNX Runtime only in this build).
 *  - Output is labeled "token_output" with output_name + shape so callers
 *    can verify whether it is logits or hidden-state.
 */

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cctype>

#if defined(CAIROS_WITH_SECURE_LOGGER)
#define CAIROS_SECURE_LOGGER 1
#include "secure_logger.h"
#else
#define CAIROS_SECURE_LOGGER 0
#endif

using cairos::GraniteBackend;
using cairos::GraniteConfig;
using cairos::GraniteEmbedder;
using cairos::GranitePrecision;
using cairos::GraniteStatus;

struct Args {
    std::string model_dir;
    std::string input_path;
    std::string output_path;
    std::string log_path;
    std::string secure_log_dir;
    std::string secure_key_path;
    std::string system_id;
    bool summary_only = false;
    std::size_t max_tokens = 0;
};

static void usage() {
    std::cerr << "Granite Hidden-State CLI\n"
              << "Usage:\n"
              << "  granite_hidden_state_cli --model-dir <dir> --input <txt> --output <jsonl>"
#if CAIROS_SECURE_LOGGER
              << " --secure-log-dir <dir> --secure-key <path>"
#else
              << " [--secure-log-dir <dir> --secure-key <path>]"
#endif
              << " [--system-id <id>] [--log <path>] [--summary-only] [--max-tokens N]\n";
}

static bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            out.model_dir = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            out.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            out.output_path = argv[++i];
        } else if (arg == "--log" && i + 1 < argc) {
            out.log_path = argv[++i];
        } else if (arg == "--secure-log-dir" && i + 1 < argc) {
            out.secure_log_dir = argv[++i];
        } else if (arg == "--secure-key" && i + 1 < argc) {
            out.secure_key_path = argv[++i];
        } else if (arg == "--system-id" && i + 1 < argc) {
            out.system_id = argv[++i];
        } else if (arg == "--summary-only") {
            out.summary_only = true;
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            out.max_tokens = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (out.model_dir.empty() || out.input_path.empty() || out.output_path.empty()) {
        return false;
    }
#if CAIROS_SECURE_LOGGER
    if (out.secure_log_dir.empty() || out.secure_key_path.empty()) {
        return false;
    }
#endif
    return true;
}
static std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

static void event_logger_cb(const char* tag, const char* detail, void* ctx) {
    auto* out = static_cast<std::ofstream*>(ctx);
    if (!out || !out->is_open()) {
        return;
    }
    (*out) << "[" << tag << "] " << detail << "\n";
    out->flush();
}


static bool load_key_bytes(const std::string& path, std::vector<uint8_t>& out_key) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    std::string data = oss.str();

    std::string hex;
    hex.reserve(data.size());
    for (unsigned char c : data) {
        if (!std::isspace(c)) {
            hex.push_back(static_cast<char>(c));
        }
    }

    auto is_hex = [](char c) {
        return std::isxdigit(static_cast<unsigned char>(c)) != 0;
    };

    bool all_hex = !hex.empty();
    for (char c : hex) {
        if (!is_hex(c)) {
            all_hex = false;
            break;
        }
    }

    if (all_hex && (hex.size() % 2 == 0)) {
        const size_t byte_len = hex.size() / 2;
        if (byte_len < 32) {
            return false;
        }
        out_key.clear();
        out_key.reserve(byte_len);
        for (size_t i = 0; i < hex.size(); i += 2) {
            std::string byte_str = hex.substr(i, 2);
            uint8_t val = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
            out_key.push_back(val);
        }
        if (out_key.size() >= 32) {
            out_key.resize(32);
            return true;
        }
        return false;
    }

    out_key.assign(data.begin(), data.end());
    if (out_key.size() < 32) {
        return false;
    }
    if (out_key.size() > 32) {
        out_key.resize(32);
    }
    return true;
}

static std::string json_kv(const std::string& key, const std::string& value) {
    return std::string("\"") + key + "\":\"" + json_escape(value) + "\"";
}

static void compute_stats(const std::vector<std::vector<float>>& tokens,
                          double& out_min,
                          double& out_max,
                          double& out_mean,
                          double& out_std) {
    double sum = 0.0;
    double sum_sq = 0.0;
    std::size_t count = 0;
    out_min = 0.0;
    out_max = 0.0;

    for (const auto& row : tokens) {
        for (float v : row) {
            const double dv = static_cast<double>(v);
            if (count == 0) {
                out_min = dv;
                out_max = dv;
            } else {
                if (dv < out_min) out_min = dv;
                if (dv > out_max) out_max = dv;
            }
            sum += dv;
            sum_sq += dv * dv;
            ++count;
        }
    }

    if (count == 0) {
        out_mean = 0.0;
        out_std = 0.0;
        return;
    }

    out_mean = sum / static_cast<double>(count);
    const double var = (sum_sq / static_cast<double>(count)) - (out_mean * out_mean);
    out_std = var > 0.0 ? std::sqrt(var) : 0.0;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage();
        return 1;
    }

    std::ifstream in(args.input_path);
    if (!in) {
        std::cerr << "Failed to open input file: " << args.input_path << "\n";
        return 1;
    }

    std::ofstream out(args.output_path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open output file: " << args.output_path << "\n";
        return 1;
    }

    std::ofstream log_file;
    if (!args.log_path.empty()) {
        log_file.open(args.log_path, std::ios::out | std::ios::trunc);
    }
    if (args.system_id.empty()) {
        args.system_id = "granite_hidden_state_cli";
    }

#if CAIROS_SECURE_LOGGER
    sl_handle* slog = nullptr;
    if (!std::filesystem::exists(args.secure_log_dir)) {
        std::filesystem::create_directories(args.secure_log_dir);
    }

    std::vector<uint8_t> master_key;
    if (!load_key_bytes(args.secure_key_path, master_key)) {
        std::cerr << "Failed to load secure logger key (need >= 32 bytes).\n";
        return 1;
    }

    sl_handle* slog = nullptr;
    sl_config_t sl_cfg{};
    sl_cfg.log_dir = args.secure_log_dir.c_str();
    sl_cfg.master_key = master_key.data();
    sl_cfg.master_key_len = master_key.size();
    sl_cfg.system_id = args.system_id.c_str();
    sl_cfg.fail_closed = 1;
    sl_cfg.key_cb = nullptr;
    sl_cfg.key_cb_ctx = nullptr;
    sl_cfg.audit_cb = nullptr;
    sl_cfg.audit_cb_ctx = nullptr;

    if (sl_init(&slog, &sl_cfg) != SL_OK) {
        std::cerr << "SecureExperimentLogger init failed.\n";
        return 2;
    }

    {
        std::ostringstream payload;
        payload << "{" << json_kv("model_dir", args.model_dir)
                << "," << json_kv("input", args.input_path)
                << "," << json_kv("output", args.output_path)
                << "," << json_kv("output_mode", args.summary_only ? "summary" : "full")
                << ",\"max_tokens\":" << args.max_tokens
                << "}";
        if (sl_log_event(slog, "granite.hidden_state.run_start", "INFO", payload.str().c_str(), "{}") != SL_OK) {
            std::cerr << "SecureExperimentLogger failed to log run_start.\n";
            sl_free(slog);
            return 3;
        }
    }

#endif

    GraniteConfig cfg;
    cfg.model_dir = args.model_dir;
    cfg.device = "cpu";
    cfg.precision = GranitePrecision::FP32;
    cfg.backend = GraniteBackend::OnnxRuntime;
    cfg.max_batch = 1;
    cfg.offline_only = true;

    GraniteEmbedder embedder(cfg);
    if (log_file.is_open()) {
        cairos::GraniteHook hook;
        hook.event_cb = event_logger_cb;
        hook.user_ctx = &log_file;
        embedder.set_hook(hook);
    }

    if (embedder.load() != GraniteStatus::Ok) {
        std::cerr << "Granite load failed: " << embedder.last_error() << "\n";
        return 2;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        lines.push_back(line);
    }

    out << std::setprecision(8);

    for (size_t i = 0; i < lines.size(); ++i) {
        std::vector<int64_t> token_ids;
        std::vector<std::vector<float>> token_embeddings;
        std::string output_name;

        GraniteStatus st = embedder.encode_tokens_raw(lines[i], token_ids, token_embeddings, output_name);
        if (st != GraniteStatus::Ok) {
            std::cerr << "Granite encode_tokens_raw failed: " << embedder.last_error() << "\n";
            return 3;
        }

        bool truncated = false;
        if (args.max_tokens > 0 && token_embeddings.size() > args.max_tokens) {
            token_embeddings.resize(args.max_tokens);
            token_ids.resize(args.max_tokens);
            truncated = true;
        }

        double min_v = 0.0, max_v = 0.0, mean_v = 0.0, std_v = 0.0;
        compute_stats(token_embeddings, min_v, max_v, mean_v, std_v);

        const std::size_t seq_len = token_embeddings.size();
        const std::size_t dim = seq_len > 0 ? token_embeddings[0].size() : 0;

        out << "{\"idx\": " << i
            << ", \"text\": \"" << json_escape(lines[i]) << "\""
            << ", \"token_ids\": [";
        for (size_t t = 0; t < token_ids.size(); ++t) {
            out << token_ids[t];
            if (t + 1 < token_ids.size()) out << ", ";
        }
        out << "]";
        out << ", \"token_count\": " << seq_len
            << ", \"dim\": " << dim
            << ", \"output_name\": \"" << json_escape(output_name) << "\""
            << ", \"output_is_logits\": " << (output_name == "logits" ? "true" : "false")
            << ", \"truncated\": " << (truncated ? "true" : "false")
            << ", \"stats\": {\"min\": " << min_v
            << ", \"max\": " << max_v
            << ", \"mean\": " << mean_v
            << ", \"std\": " << std_v << "}";

        if (!args.summary_only) {
            out << ", \"token_output\": [";
            for (size_t t = 0; t < token_embeddings.size(); ++t) {
                out << "[";
                const auto& row = token_embeddings[t];
                for (size_t d = 0; d < row.size(); ++d) {
                    out << row[d];
                    if (d + 1 < row.size()) out << ", ";
                }
                out << "]";
                if (t + 1 < token_embeddings.size()) out << ", ";
            }
            out << "]";
        }
        out << "}\n";

#if CAIROS_SECURE_LOGGER
        {
            std::ostringstream payload;
            payload << "{"
                    << "\"idx\":" << i
                    << "," << json_kv("output_name", output_name)
                    << ",\"token_count\":" << seq_len
                    << ",\"dim\":" << dim
                    << ",\"min\":" << min_v
                    << ",\"max\":" << max_v
                    << ",\"mean\":" << mean_v
                    << ",\"std\":" << std_v
                    << ",\"truncated\":" << (truncated ? "true" : "false")
                    << "}";
            if (sl_log_event(slog, "granite.hidden_state.line", "INFO", payload.str().c_str(), "{}") != SL_OK) {
                std::cerr << "SecureExperimentLogger failed to log line.\n";
                sl_free(slog);
                return 4;
            }
        }
#endif
    }

#if CAIROS_SECURE_LOGGER
    if (sl_log_event(slog, "granite.hidden_state.run_end", "INFO", "{}", "{}") != SL_OK) {
        std::cerr << "SecureExperimentLogger failed to log run_end.\n";
        sl_free(slog);
        return 5;
    }
    sl_free(slog);
#endif

    embedder.unload();
    return 0;
}
