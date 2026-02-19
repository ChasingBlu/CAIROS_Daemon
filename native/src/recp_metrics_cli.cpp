#include "granite_embedder.hpp"
#include "recp_metrics.hpp"
#if defined(CAIROS_WITH_SECURE_LOGGER)
#define CAIROS_SECURE_LOGGER 1
#include "secure_logger.h"
#else
#define CAIROS_SECURE_LOGGER 0
#endif

#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using cairos::GraniteBackend;
using cairos::GraniteConfig;
using cairos::GraniteEmbedder;
using cairos::GranitePrecision;
using cairos::GraniteStatus;
using cairos::RecpMetrics;

struct Args {
    std::string model_dir;
    std::string turns_path;
    std::string embeddings_path;
    std::string token_counts_path;
    std::string anchors_path;
    std::string output_path;
    std::string log_path;
    std::string secure_log_dir;
    std::string secure_key_path;
    std::string system_id;
    std::size_t srv_k = 1;
};

static void usage() {
    std::cerr << "RECP Metrics CLI\n"
              << "Usage:\n"
              << "  recp_metrics_cli --turns <txt> --output <json>"
#if CAIROS_SECURE_LOGGER
              << " --secure-log-dir <dir> --secure-key <path>"
#else
              << " [--secure-log-dir <dir> --secure-key <path>]"
#endif
              << " [--anchors <txt>] [--embeddings <jsonl> | --model-dir <dir>] [--token-counts <jsonl>]"
              << " [--srv-k N] [--log <path>] [--system-id <id>]\n";
}

static bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            out.model_dir = argv[++i];
        } else if (arg == "--turns" && i + 1 < argc) {
            out.turns_path = argv[++i];
        } else if (arg == "--embeddings" && i + 1 < argc) {
            out.embeddings_path = argv[++i];
        } else if (arg == "--token-counts" && i + 1 < argc) {
            out.token_counts_path = argv[++i];
        } else if (arg == "--anchors" && i + 1 < argc) {
            out.anchors_path = argv[++i];
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
        } else if (arg == "--srv-k" && i + 1 < argc) {
            out.srv_k = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (out.turns_path.empty() || out.output_path.empty()) {
        return false;
    }
#if CAIROS_SECURE_LOGGER
    if (out.secure_log_dir.empty() || out.secure_key_path.empty()) {
        return false;
    }
#endif
    if (out.model_dir.empty() && out.embeddings_path.empty()) {
        std::cerr << "Either --model-dir or --embeddings must be provided.\n";
        return false;
    }
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

static bool read_lines(const std::string& path, std::vector<std::string>& out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            out.push_back(line);
        }
    }
    return true;
}

static bool parse_embedding_line(const std::string& line, std::vector<float>& out) {
    const std::string key = "\"embedding\"";
    auto key_pos = line.find(key);
    if (key_pos == std::string::npos) {
        return false;
    }
    auto open_pos = line.find('[', key_pos);
    auto close_pos = line.find(']', open_pos);
    if (open_pos == std::string::npos || close_pos == std::string::npos || close_pos <= open_pos) {
        return false;
    }
    std::string inner = line.substr(open_pos + 1, close_pos - open_pos - 1);
    std::vector<float> values;
    values.reserve(1024);
    std::stringstream ss(inner);
    std::string token;
    while (std::getline(ss, token, ',')) {
        std::stringstream conv(token);
        float val = 0.0f;
        conv >> val;
        if (!conv.fail()) {
            values.push_back(val);
        }
    }
    if (values.empty()) {
        return false;
    }
    out = std::move(values);
    return true;
}

static bool read_embeddings_jsonl(const std::string& path, std::vector<std::vector<float>>& out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<float> emb;
        if (!parse_embedding_line(line, emb)) {
            return false;
        }
        out.push_back(std::move(emb));
    }
    return true;
}

static std::size_t word_count(const std::string& text) {
    std::size_t count = 0;
    bool in_word = false;
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            if (in_word) {
                in_word = false;
            }
        } else if (!in_word) {
            in_word = true;
            ++count;
        }
    }
    return count;
}

static bool parse_token_count_line(const std::string& line, int& out_count) {
    const std::string key = "\"token_count\"";
    auto key_pos = line.find(key);
    if (key_pos == std::string::npos) {
        return false;
    }
    auto colon = line.find(':', key_pos + key.size());
    if (colon == std::string::npos) {
        return false;
    }
    auto start = line.find_first_of("-0123456789", colon);
    if (start == std::string::npos) {
        return false;
    }
    auto end = line.find_first_not_of("0123456789", start);
    std::string num = line.substr(start, end - start);
    try {
        out_count = std::stoi(num);
    } catch (...) {
        return false;
    }
    return true;
}

static bool read_token_counts_jsonl(const std::string& path, std::vector<int>& out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        int count = 0;
        if (!parse_token_count_line(line, count)) {
            return false;
        }
        out.push_back(count);
    }
    return true;
}

static double token_to_word_variance_counts(const std::vector<std::string>& turns,
                                            const std::vector<int>& token_counts) {
    std::vector<double> ratios;
    ratios.reserve(turns.size());
    for (std::size_t i = 0; i < turns.size() && i < token_counts.size(); ++i) {
        std::size_t wc = word_count(turns[i]);
        if (wc == 0) {
            continue;
        }
        ratios.push_back(static_cast<double>(token_counts[i]) / static_cast<double>(wc));
    }
    if (ratios.empty()) {
        return 0.0;
    }
    double mean = 0.0;
    for (double v : ratios) {
        mean += v;
    }
    mean /= static_cast<double>(ratios.size());
    double var = 0.0;
    for (double v : ratios) {
        double diff = v - mean;
        var += diff * diff;
    }
    var /= static_cast<double>(ratios.size());
    return var;
}

static std::string json_kv(const std::string& key, const std::string& value) {
    return std::string("\"") + key + "\":\"" + json_escape(value) + "\"";
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage();
        return 1;
    }

    std::ofstream log_file;
    if (!args.log_path.empty()) {
        log_file.open(args.log_path, std::ios::out | std::ios::trunc);
    }

#if CAIROS_SECURE_LOGGER
    sl_handle* slog = nullptr;
    // SecureExperimentLogger (fail-closed)
    std::vector<uint8_t> key;
    if (!load_key_bytes(args.secure_key_path, key)) {
        std::cerr << "Failed to load secure key: " << args.secure_key_path << "\n";
        return 2;
    }

    sl_config_t cfg{};
    cfg.log_dir = args.secure_log_dir.c_str();
    cfg.master_key = key.data();
    cfg.master_key_len = key.size();
    cfg.system_id = args.system_id.empty() ? "recp_metrics_cli" : args.system_id.c_str();
    cfg.fail_closed = 1;

    sl_status_t slst = sl_init(&slog, &cfg);
    if (slst != SL_OK || slog == nullptr) {
        std::cerr << "Secure logger init failed (status " << slst << ")\n";
        return 3;
    }
#endif

    std::vector<std::string> turns;
    if (!read_lines(args.turns_path, turns) || turns.empty()) {
        std::cerr << "Failed to read turns: " << args.turns_path << "\n";
#if CAIROS_SECURE_LOGGER
        sl_free(slog);
#endif
        return 4;
    }

    std::vector<std::string> anchors;
    if (!args.anchors_path.empty()) {
        if (!read_lines(args.anchors_path, anchors)) {
            std::cerr << "Failed to read anchors: " << args.anchors_path << "\n";
    #if CAIROS_SECURE_LOGGER
        sl_free(slog);
#endif
            return 5;
        }
    }

    std::vector<std::vector<float>> embeddings;
    std::vector<std::vector<float>> anchor_embeddings;
    std::string embeddings_source;
    if (!args.embeddings_path.empty()) {
        if (!read_embeddings_jsonl(args.embeddings_path, embeddings)) {
            std::cerr << "Failed to read embeddings JSONL: " << args.embeddings_path << "\n";
    #if CAIROS_SECURE_LOGGER
        sl_free(slog);
#endif
            return 6;
        }
        embeddings_source = args.embeddings_path;
    } else {
        GraniteConfig gcfg;
        gcfg.model_dir = args.model_dir;
        gcfg.device = "cpu";
        gcfg.precision = GranitePrecision::FP32;
        gcfg.backend = GraniteBackend::OnnxRuntime;
        gcfg.max_batch = 1;
        gcfg.offline_only = true;

        GraniteEmbedder embedder(gcfg);
        if (log_file.is_open()) {
            cairos::GraniteHook hook;
            hook.event_cb = event_logger_cb;
            hook.user_ctx = &log_file;
            embedder.set_hook(hook);
        }

        if (embedder.load() != GraniteStatus::Ok) {
            std::cerr << "Granite load failed: " << embedder.last_error() << "\n";
    #if CAIROS_SECURE_LOGGER
        sl_free(slog);
#endif
            return 7;
        }
        GraniteStatus st = embedder.encode(turns, embeddings);
        if (st != GraniteStatus::Ok) {
            std::cerr << "Granite encode failed: " << embedder.last_error() << "\n";
            embedder.unload();
    #if CAIROS_SECURE_LOGGER
        sl_free(slog);
#endif
            return 8;
        }
        embedder.unload();
        embeddings_source = args.model_dir;
    }


    if (!anchors.empty()) {
        if (!args.model_dir.empty()) {
            GraniteConfig acfg;
            acfg.model_dir = args.model_dir;
            acfg.device = "cpu";
            acfg.precision = GranitePrecision::FP32;
            acfg.backend = GraniteBackend::OnnxRuntime;
            acfg.max_batch = 1;
            acfg.offline_only = true;

            GraniteEmbedder anchor_embedder(acfg);
            if (log_file.is_open()) {
                cairos::GraniteHook hook;
                hook.event_cb = event_logger_cb;
                hook.user_ctx = &log_file;
                anchor_embedder.set_hook(hook);
            }

            if (anchor_embedder.load() != GraniteStatus::Ok) {
                std::cerr << "Granite load failed for anchors: " << anchor_embedder.last_error() << "\n";
        #if CAIROS_SECURE_LOGGER
        sl_free(slog);
#endif
                return 7;
            }
            GraniteStatus ast = anchor_embedder.encode(anchors, anchor_embeddings);
            if (ast != GraniteStatus::Ok) {
                std::cerr << "Granite encode failed for anchors: " << anchor_embedder.last_error() << "\n";
                anchor_embedder.unload();
        #if CAIROS_SECURE_LOGGER
        sl_free(slog);
#endif
                return 8;
            }
            anchor_embedder.unload();
        } else {
            std::cerr << "Anchors provided but --model-dir missing; ICS will use legacy pairwise fallback.\n";
        }
    }

    if (embeddings.size() != turns.size()) {
        std::cerr << "Embedding count mismatch: " << embeddings.size() << " vs " << turns.size() << "\n";
#if CAIROS_SECURE_LOGGER
        sl_free(slog);
#endif
        return 9;
    }

    std::string tokenization_source = "legacy_char_word";
    std::vector<int> token_counts;
    const std::vector<int>* token_counts_ptr = nullptr;
    if (!args.token_counts_path.empty()) {
        if (!read_token_counts_jsonl(args.token_counts_path, token_counts)) {
            std::cerr << "Failed to read token counts JSONL: " << args.token_counts_path << "\n";
            return 9;
        }
        if (token_counts.size() != turns.size()) {
            std::cerr << "Token counts mismatch: " << token_counts.size() << " vs " << turns.size() << "\n";
            return 9;
        }
        tokenization_source = args.token_counts_path;
        token_counts_ptr = &token_counts;
    }

    RecpMetrics metrics = cairos::calculate_recp_metrics(turns, embeddings, anchors, anchor_embeddings, args.srv_k, token_counts_ptr);


    std::ofstream out(args.output_path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open output: " << args.output_path << "\n";
#if CAIROS_SECURE_LOGGER
        sl_free(slog);
#endif
        return 10;
    }

    out << std::setprecision(8);
    out << "{\n";
    out << "  \"turns_count\": " << turns.size() << ",\n";
    out << "  \"anchors_count\": " << anchors.size() << ",\n";
    out << "  \"srv_k\": " << args.srv_k << ",\n";
    out << "  \"metrics\": {\n";
    out << "    \"ICS\": " << metrics.ics << ",\n";
    out << "    \"ICS_pairwise\": " << metrics.ics_pairwise << ",\n";
    if (metrics.ics_anchor_available) {
        out << "    \"ICS_anchor_centroid\": " << metrics.ics_anchor << ",\n";
    } else {
        out << "    \"ICS_anchor_centroid\": null,\n";
    }
    out << "    \"API\": " << metrics.api << ",\n";
    out << "    \"ECR\": " << metrics.ecr << ",\n";
    out << "    \"LDI\": " << metrics.ldi << ",\n";
    out << "    \"TCDM\": " << metrics.twv << ",\n";
    out << "    \"SRV\": " << metrics.srv << ",\n";
    out << "    \"HMV\": " << metrics.hmv << ",\n";
    out << "    \"RDS\": " << metrics.rds << ",\n";
    out << "    \"MPI\": " << metrics.mpi << ",\n";
    out << "    \"SDA\": " << metrics.sda << ",\n";
    out << "    \"ASV\": " << metrics.asv << "\n";
    out << "  },\n";
    out << "  \"availability\": {\n";
    out << "    \"SDA\": " << (metrics.sda_available ? "true" : "false") << ",\n";
    out << "    \"ASV\": " << (metrics.asv_available ? "true" : "false") << "\n";
    out << "  },\n";
    out << "  \"embedding_source\": \"" << json_escape(embeddings_source) << "\"\n";
    out << "}\n";

    // Secure log payload
    std::ostringstream payload;
    payload << "{";
    payload << "\"ICS\":" << metrics.ics << ",";
    payload << "\"ICS_pairwise\":" << metrics.ics_pairwise << ",";
    payload << "\"API\":" << metrics.api << ",";
    payload << "\"ECR\":" << metrics.ecr << ",";
    payload << "\"LDI\":" << metrics.ldi << ",";
    payload << "\"TCDM\":" << metrics.twv << ",";
    payload << "\"SRV\":" << metrics.srv << ",";
    payload << "\"HMV\":" << metrics.hmv << ",";
    payload << "\"RDS\":" << metrics.rds << ",";
    payload << "\"MPI\":" << metrics.mpi << ",";
    payload << "\"SDA\":" << metrics.sda << ",";
    payload << "\"ASV\":" << metrics.asv << ",";
    payload << "\"SDA_available\":" << (metrics.sda_available ? "true" : "false") << ",";
    payload << "\"ASV_available\":" << (metrics.asv_available ? "true" : "false") << "";
    payload << "}";

    std::ostringstream metadata;
    metadata << "{";
    metadata << json_kv("turns_path", args.turns_path) << ",";
    if (!args.anchors_path.empty()) {
        metadata << json_kv("anchors_path", args.anchors_path) << ",";
    }
    metadata << json_kv("embedding_source", embeddings_source) << ",";
    metadata << json_kv("tokenization_source", tokenization_source) << ",";
    metadata << "\"srv_k\":" << args.srv_k;
    metadata << "}";

#if CAIROS_SECURE_LOGGER
    sl_log_event(slog, "recp.metrics", "INFO", payload.str().c_str(), metadata.str().c_str());
    sl_verify(slog);
    sl_free(slog);
#endif

    return 0;
}

