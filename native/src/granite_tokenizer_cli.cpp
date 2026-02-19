#include "granite_embedder.hpp"

#include <cctype>
#include <fstream>
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
    bool add_bos_eos = true;
};

static void usage() {
    std::cerr << "Granite Tokenizer CLI (model-consistent)\n"
              << "Usage:\n"
              << "  granite_tokenizer_cli --model-dir <dir> --input <txt> --output <jsonl>"
              << " [--no-bos-eos] [--log <path>]\n";
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
        } else if (arg == "--no-bos-eos") {
            out.add_bos_eos = false;
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

static void event_logger_cb(const char* tag, const char* detail, void* ctx) {
    auto* out = static_cast<std::ofstream*>(ctx);
    if (!out || !out->is_open()) {
        return;
    }
    (*out) << "[" << tag << "] " << detail << "\n";
    out->flush();
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

static std::size_t word_count(const std::string& text) {
    std::istringstream iss(text);
    std::size_t count = 0;
    std::string token;
    while (iss >> token) {
        ++count;
    }
    return count;
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

    GraniteStatus st = embedder.load();
    if (st != GraniteStatus::Ok) {
        std::cerr << "Granite load failed: " << embedder.last_error() << "\n";
        return 2;
    }

    std::vector<std::string> lines;
    if (!read_lines(args.input_path, lines) || lines.empty()) {
        std::cerr << "Failed to read input: " << args.input_path << "\n";
        embedder.unload();
        return 3;
    }

    std::ofstream out(args.output_path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open output: " << args.output_path << "\n";
        embedder.unload();
        return 4;
    }

    for (std::size_t i = 0; i < lines.size(); ++i) {
        std::vector<int64_t> token_ids;
        GraniteStatus ts = embedder.tokenize(lines[i], token_ids, args.add_bos_eos);
        if (ts != GraniteStatus::Ok) {
            std::cerr << "Tokenize failed at line " << i << ": " << embedder.last_error() << "\n";
            embedder.unload();
            return 5;
        }
        const std::size_t wcount = word_count(lines[i]);
        out << "{\"idx\":" << i
            << ",\"token_count\":" << token_ids.size()
            << ",\"word_count\":" << wcount
            << "}\n";
    }

    embedder.unload();
    return 0;
}
