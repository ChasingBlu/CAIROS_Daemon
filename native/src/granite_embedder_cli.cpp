#include "granite_embedder.hpp"

/*
 * Granite Embedder CLI (C++)
 *
 * Purpose:
 *  - Minimal, auditable CLI wrapper around GraniteEmbedder.
 *  - Produces JSONL outputs aligned with Python loader artifacts.
 *
 * REPA/ISO notes:
 *  - No silent fallback: exits non-zero on errors.
 *  - Explicit backend selection (ONNX Runtime only in this build).
 *  - Optional event hook emits to log file for SecureExperimentLogger integration.
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
};

static void usage() {
    std::cerr << "Granite Embedder CLI\n"
              << "Usage:\n"
              << "  granite_embedder_cli --model-dir <dir> --input <txt> --output <jsonl> [--log <path>]\n";
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

    std::vector<std::vector<float>> embeddings;
    GraniteStatus st = embedder.encode(lines, embeddings);
    if (st != GraniteStatus::Ok) {
        std::cerr << "Granite encode failed: " << embedder.last_error() << "\n";
        return 3;
    }

    if (embeddings.size() != lines.size()) {
        std::cerr << "Embedding count mismatch: " << embeddings.size() << " vs " << lines.size() << "\n";
        return 4;
    }

    out << std::setprecision(8);
    for (size_t i = 0; i < lines.size(); ++i) {
        out << "{\"idx\": " << i << ", \"text\": \"" << json_escape(lines[i]) << "\", \"embedding\": [";
        const auto& emb = embeddings[i];
        for (size_t j = 0; j < emb.size(); ++j) {
            out << emb[j];
            if (j + 1 < emb.size()) {
                out << ", ";
            }
        }
        out << "]}\n";
    }

    embedder.unload();
    return 0;
}
