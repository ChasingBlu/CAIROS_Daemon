#include "granite_embedder.hpp"
#include "identity_anchor.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using cairos::AnchorMethod;
using cairos::GraniteBackend;
using cairos::GraniteConfig;
using cairos::GraniteEmbedder;
using cairos::GranitePrecision;
using cairos::GraniteStatus;
using cairos::IdentityAnchorConfig;
using cairos::IdentityAnchorDiagnostics;

struct Args {
    std::string model_dir;
    std::string anchors_path;
    std::string weights_path;
    std::string prompt_path;
    std::string output_path;
    std::string method;
    bool allow_fallback = false;
    bool normalize_output = true;
};

static void usage() {
    std::cerr << "Identity Anchor CLI\n"
              << "Usage:\n"
              << "  identity_anchor_cli --model-dir <dir> --anchors <txt> --output <json>"
              << " --method <static|weighted|context|context-mean|pca> [--weights <txt>]"
              << " [--prompt <txt>] [--allow-fallback] [--no-normalize]\n";
}

static bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-dir" && i + 1 < argc) {
            out.model_dir = argv[++i];
        } else if (arg == "--anchors" && i + 1 < argc) {
            out.anchors_path = argv[++i];
        } else if (arg == "--weights" && i + 1 < argc) {
            out.weights_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            out.prompt_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            out.output_path = argv[++i];
        } else if (arg == "--method" && i + 1 < argc) {
            out.method = argv[++i];
        } else if (arg == "--allow-fallback") {
            out.allow_fallback = true;
        } else if (arg == "--no-normalize") {
            out.normalize_output = false;
        } else if (arg == "--help" || arg == "-h") {
            return false;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    if (out.model_dir.empty() || out.anchors_path.empty() || out.output_path.empty() || out.method.empty()) {
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

static bool read_weights(const std::string& path, std::vector<float>& out) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        out.push_back(std::stof(line));
    }
    return true;
}

static AnchorMethod parse_method(const std::string& m) {
    if (m == "static") return AnchorMethod::StaticCentroid;
    if (m == "weighted") return AnchorMethod::WeightedCentroid;
    if (m == "context") return AnchorMethod::ContextualTokenPooling;
    if (m == "context-mean") return AnchorMethod::ContextualPromptMean;
    if (m == "pca") return AnchorMethod::PCAFirstComponent;
    return AnchorMethod::StaticCentroid;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        usage();
        return 1;
    }

    std::vector<std::string> anchors;
    if (!read_lines(args.anchors_path, anchors) || anchors.empty()) {
        std::cerr << "Failed to read anchors from: " << args.anchors_path << "\n";
        return 2;
    }

    std::vector<float> weights;
    if (!args.weights_path.empty()) {
        if (!read_weights(args.weights_path, weights)) {
            std::cerr << "Failed to read weights from: " << args.weights_path << "\n";
            return 3;
        }
    }

    IdentityAnchorConfig cfg;
    cfg.method = parse_method(args.method);
    cfg.allow_fallback_prompt_mean = args.allow_fallback;
    cfg.normalize_output = args.normalize_output;

    if (!args.prompt_path.empty()) {
        std::ifstream prompt_in(args.prompt_path);
        if (!prompt_in) {
            std::cerr << "Failed to read prompt from: " << args.prompt_path << "\n";
            return 4;
        }
        std::ostringstream buffer;
        buffer << prompt_in.rdbuf();
        cfg.prompt_template = buffer.str();
    }

    GraniteConfig gcfg;
    gcfg.model_dir = args.model_dir;
    gcfg.device = "cpu";
    gcfg.precision = GranitePrecision::FP32;
    gcfg.backend = GraniteBackend::OnnxRuntime;
    gcfg.max_batch = 1;
    gcfg.offline_only = true;

    GraniteEmbedder embedder(gcfg);
    if (embedder.load() != GraniteStatus::Ok) {
        std::cerr << "Granite load failed: " << embedder.last_error() << "\n";
        return 5;
    }

    IdentityAnchorDiagnostics diag;
    std::vector<float> anchor_vec;
    const std::vector<float>* weight_ptr = weights.empty() ? nullptr : &weights;
    GraniteStatus st = cairos::build_identity_anchor(embedder, anchors, weight_ptr, cfg, anchor_vec, &diag);
    if (st != GraniteStatus::Ok) {
        std::cerr << "Identity anchor build failed: " << embedder.last_error() << "\n";
        if (!diag.warnings.empty()) {
            for (const auto& w : diag.warnings) {
                std::cerr << "WARN: " << w << "\n";
            }
        }
        return 6;
    }

    std::ofstream out(args.output_path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open output file: " << args.output_path << "\n";
        return 7;
    }

    out << std::setprecision(8);
    out << "{\n";
    out << "  \"method\": \"" << json_escape(diag.method_used) << "\",\n";
    out << "  \"anchors\": [";
    for (size_t i = 0; i < anchors.size(); ++i) {
        out << "\"" << json_escape(anchors[i]) << "\"";
        if (i + 1 < anchors.size()) out << ", ";
    }
    out << "],\n";
    if (!weights.empty()) {
        out << "  \"weights\": [";
        for (size_t i = 0; i < weights.size(); ++i) {
            out << weights[i];
            if (i + 1 < weights.size()) out << ", ";
        }
        out << "],\n";
    }
    out << "  \"allow_fallback\": " << (cfg.allow_fallback_prompt_mean ? "true" : "false") << ",\n";
    out << "  \"normalize_output\": " << (cfg.normalize_output ? "true" : "false") << ",\n";
    out << "  \"matched_token_count\": " << diag.matched_token_count << ",\n";
    out << "  \"anchor_token_count\": " << diag.anchor_token_count << ",\n";
    out << "  \"used_fallback\": " << (diag.used_fallback ? "true" : "false") << ",\n";

    out << "  \"warnings\": [";
    for (size_t i = 0; i < diag.warnings.size(); ++i) {
        out << "\"" << json_escape(diag.warnings[i]) << "\"";
        if (i + 1 < diag.warnings.size()) out << ", ";
    }
    out << "],\n";

    out << "  \"notes\": [";
    for (size_t i = 0; i < diag.notes.size(); ++i) {
        out << "\"" << json_escape(diag.notes[i]) << "\"";
        if (i + 1 < diag.notes.size()) out << ", ";
    }
    out << "],\n";

    out << "  \"vector\": [";
    for (size_t i = 0; i < anchor_vec.size(); ++i) {
        out << anchor_vec[i];
        if (i + 1 < anchor_vec.size()) out << ", ";
    }
    out << "]\n";
    out << "}\n";

    embedder.unload();
    return 0;
}
