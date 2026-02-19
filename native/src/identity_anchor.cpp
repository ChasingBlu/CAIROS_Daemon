#include "identity_anchor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace {

void add_warning(cairos::IdentityAnchorDiagnostics* diag, const std::string& msg) {
    if (diag) {
        diag->warnings.push_back(msg);
    }
}

void add_note(cairos::IdentityAnchorDiagnostics* diag, const std::string& msg) {
    if (diag) {
        diag->notes.push_back(msg);
    }
}

std::string join_anchors(const std::vector<std::string>& anchors) {
    std::ostringstream oss;
    for (size_t i = 0; i < anchors.size(); ++i) {
        oss << anchors[i];
        if (i + 1 < anchors.size()) {
            oss << ", ";
        }
    }
    return oss.str();
}

std::string build_prompt(const std::vector<std::string>& anchors, const std::string& tpl) {
    const std::string token = "{anchors}";
    std::string prompt = tpl;
    const auto pos = prompt.find(token);
    if (pos != std::string::npos) {
        prompt.replace(pos, token.size(), join_anchors(anchors));
    } else {
        prompt += " ";
        prompt += join_anchors(anchors);
        prompt += ".";
    }
    return prompt;
}

bool normalize_vector(std::vector<float>& v) {
    double sum = 0.0;
    for (float x : v) {
        sum += static_cast<double>(x) * static_cast<double>(x);
    }
    if (sum <= 0.0) {
        return false;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(sum));
    for (auto& x : v) {
        x *= inv;
    }
    return true;
}

std::vector<float> mean_pool(const std::vector<std::vector<float>>& vectors) {
    if (vectors.empty()) {
        return {};
    }
    const size_t dim = vectors[0].size();
    std::vector<float> out(dim, 0.0f);
    for (const auto& v : vectors) {
        if (v.size() != dim) {
            return {};
        }
        for (size_t i = 0; i < dim; ++i) {
            out[i] += v[i];
        }
    }
    const float inv = 1.0f / static_cast<float>(vectors.size());
    for (auto& x : out) {
        x *= inv;
    }
    return out;
}

std::vector<float> weighted_pool(const std::vector<std::vector<float>>& vectors,
                                 const std::vector<float>& weights) {
    if (vectors.empty()) {
        return {};
    }
    if (vectors.size() != weights.size()) {
        return {};
    }
    const size_t dim = vectors[0].size();
    std::vector<float> out(dim, 0.0f);
    double sum_w = 0.0;
    for (float w : weights) {
        sum_w += w;
    }
    if (sum_w <= 0.0) {
        return {};
    }
    for (size_t j = 0; j < vectors.size(); ++j) {
        const auto& v = vectors[j];
        if (v.size() != dim) {
            return {};
        }
        const float w = weights[j] / static_cast<float>(sum_w);
        for (size_t i = 0; i < dim; ++i) {
            out[i] += v[i] * w;
        }
    }
    return out;
}

std::vector<float> pca_first_component(const std::vector<std::vector<float>>& samples,
                                       int iterations) {
    if (samples.size() < 2) {
        return {};
    }
    const size_t dim = samples[0].size();
    if (dim == 0) {
        return {};
    }
    for (const auto& v : samples) {
        if (v.size() != dim) {
            return {};
        }
    }

    std::vector<float> mean(dim, 0.0f);
    for (const auto& v : samples) {
        for (size_t i = 0; i < dim; ++i) {
            mean[i] += v[i];
        }
    }
    const float inv_n = 1.0f / static_cast<float>(samples.size());
    for (auto& m : mean) {
        m *= inv_n;
    }

    std::vector<std::vector<float>> centered(samples.size(), std::vector<float>(dim, 0.0f));
    for (size_t j = 0; j < samples.size(); ++j) {
        for (size_t i = 0; i < dim; ++i) {
            centered[j][i] = samples[j][i] - mean[i];
        }
    }

    std::vector<float> v = centered[0];
    if (!normalize_vector(v)) {
        return {};
    }

    for (int it = 0; it < iterations; ++it) {
        std::vector<float> w(dim, 0.0f);
        for (const auto& x : centered) {
            double dot = 0.0;
            for (size_t i = 0; i < dim; ++i) {
                dot += static_cast<double>(x[i]) * static_cast<double>(v[i]);
            }
            for (size_t i = 0; i < dim; ++i) {
                w[i] += static_cast<float>(dot) * x[i];
            }
        }
        if (!normalize_vector(w)) {
            return {};
        }
        v.swap(w);
    }

    return v;
}

void find_anchor_positions(const std::vector<int64_t>& prompt_ids,
                           const std::vector<int64_t>& anchor_ids,
                           std::vector<char>& is_anchor) {
    if (anchor_ids.empty() || prompt_ids.empty()) {
        return;
    }
    const size_t max_start = (prompt_ids.size() >= anchor_ids.size())
        ? (prompt_ids.size() - anchor_ids.size())
        : 0;

    for (size_t i = 0; i <= max_start; ++i) {
        bool match = true;
        for (size_t j = 0; j < anchor_ids.size(); ++j) {
            if (prompt_ids[i + j] != anchor_ids[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            for (size_t j = 0; j < anchor_ids.size(); ++j) {
                is_anchor[i + j] = 1;
            }
        }
    }
}

} // namespace

namespace cairos {

GraniteStatus build_identity_anchor(GraniteEmbedder& embedder,
                                    const std::vector<std::string>& anchors,
                                    const std::vector<float>* weights,
                                    const IdentityAnchorConfig& cfg,
                                    std::vector<float>& out_vector,
                                    IdentityAnchorDiagnostics* diag) {
    if (anchors.empty()) {
        if (diag) {
            diag->method_used = "error";
        }
        return GraniteStatus::ErrArg;
    }

    if (!embedder.is_loaded()) {
        if (diag) {
            diag->method_used = "error";
        }
        return GraniteStatus::ErrState;
    }

    if (diag) {
        diag->method_used.clear();
        diag->warnings.clear();
        diag->notes.clear();
        diag->matched_token_count = 0;
        diag->anchor_token_count = 0;
        diag->used_fallback = false;
    }

    const std::string prompt = build_prompt(anchors, cfg.prompt_template);

    if (cfg.method == AnchorMethod::ContextualPromptMean) {
        std::vector<std::vector<float>> prompt_embeddings;
        GraniteStatus st = embedder.encode({prompt}, prompt_embeddings);
        if (st != GraniteStatus::Ok || prompt_embeddings.empty()) {
            add_warning(diag, "Contextual prompt mean failed; embedder.encode returned error.");
            return st;
        }
        out_vector = prompt_embeddings[0];
        if (cfg.normalize_output && !normalize_vector(out_vector)) {
            add_warning(diag, "Prompt mean normalization failed (zero norm). Returning raw vector.");
        }
        if (diag) {
            diag->method_used = "ContextualPromptMean";
        }
        return GraniteStatus::Ok;
    }

    if (cfg.method == AnchorMethod::ContextualTokenPooling) {
        std::vector<int64_t> prompt_token_ids;
        std::vector<std::vector<float>> prompt_token_embeddings;

        GraniteStatus st = embedder.encode_tokens(prompt, prompt_token_ids, prompt_token_embeddings);
        if (st != GraniteStatus::Ok) {
            if (cfg.allow_fallback_prompt_mean) {
                add_warning(diag, "Token pooling unavailable; falling back to prompt mean (opt-in).");
                if (diag) {
                    diag->used_fallback = true;
                }
                IdentityAnchorConfig fallback_cfg = cfg;
                fallback_cfg.method = AnchorMethod::ContextualPromptMean;
                return build_identity_anchor(embedder, anchors, weights, fallback_cfg, out_vector, diag);
            }
            return st;
        }

        std::vector<char> is_anchor(prompt_token_ids.size(), 0);
        std::size_t anchor_token_total = 0;

        for (const auto& anchor : anchors) {
            std::vector<int64_t> anchor_ids;
            GraniteStatus tok_st = embedder.tokenize(anchor, anchor_ids, false);
            if (tok_st != GraniteStatus::Ok) {
                add_warning(diag, "Anchor tokenization failed for: " + anchor);
                continue;
            }
            anchor_token_total += anchor_ids.size();
            if (anchor_ids.empty()) {
                add_warning(diag, "Anchor token list empty for: " + anchor);
                continue;
            }
            find_anchor_positions(prompt_token_ids, anchor_ids, is_anchor);
        }

        std::size_t matched = 0;
        for (char flag : is_anchor) {
            if (flag) {
                matched++;
            }
        }

        if (diag) {
            diag->anchor_token_count = anchor_token_total;
            diag->matched_token_count = matched;
        }

        if (matched == 0) {
            if (cfg.allow_fallback_prompt_mean) {
                add_warning(diag, "No anchor tokens matched; falling back to prompt mean (opt-in).");
                if (diag) {
                    diag->used_fallback = true;
                }
                IdentityAnchorConfig fallback_cfg = cfg;
                fallback_cfg.method = AnchorMethod::ContextualPromptMean;
                return build_identity_anchor(embedder, anchors, weights, fallback_cfg, out_vector, diag);
            }
            add_warning(diag, "No anchor tokens matched; no fallback permitted.");
            return GraniteStatus::ErrArg;
        }

        const size_t dim = prompt_token_embeddings[0].size();
        std::vector<float> pooled(dim, 0.0f);
        std::size_t pooled_count = 0;

        for (size_t i = 0; i < is_anchor.size(); ++i) {
            if (!is_anchor[i]) {
                continue;
            }
            const auto& token_vec = prompt_token_embeddings[i];
            if (token_vec.size() != dim) {
                add_warning(diag, "Token embedding dimension mismatch during pooling.");
                return GraniteStatus::ErrBackend;
            }
            for (size_t d = 0; d < dim; ++d) {
                pooled[d] += token_vec[d];
            }
            pooled_count += 1;
        }

        if (pooled_count == 0) {
            add_warning(diag, "Token pooling resulted in zero matched tokens.");
            return GraniteStatus::ErrBackend;
        }

        const float inv = 1.0f / static_cast<float>(pooled_count);
        for (auto& v : pooled) {
            v *= inv;
        }

        out_vector = pooled;
        if (cfg.normalize_output && !normalize_vector(out_vector)) {
            add_warning(diag, "Token pooling normalization failed (zero norm). Returning raw vector.");
        }
        if (diag) {
            diag->method_used = "ContextualTokenPooling";
        }
        return GraniteStatus::Ok;
    }

    std::vector<std::vector<float>> anchor_embeddings;
    GraniteStatus st = embedder.encode(anchors, anchor_embeddings);
    if (st != GraniteStatus::Ok) {
        add_warning(diag, "Anchor embedding encode failed.");
        return st;
    }

    if (cfg.method == AnchorMethod::WeightedCentroid) {
        if (!weights || weights->empty()) {
            add_warning(diag, "Weighted centroid requested but no weights provided.");
            return GraniteStatus::ErrArg;
        }
        out_vector = weighted_pool(anchor_embeddings, *weights);
        if (out_vector.empty()) {
            add_warning(diag, "Weighted centroid failed (dimension mismatch or zero weights)." );
            return GraniteStatus::ErrArg;
        }
        if (cfg.normalize_output && !normalize_vector(out_vector)) {
            add_warning(diag, "Weighted centroid normalization failed (zero norm). Returning raw vector.");
        }
        if (diag) {
            diag->method_used = "WeightedCentroid";
        }
        return GraniteStatus::Ok;
    }

    if (cfg.method == AnchorMethod::PCAFirstComponent) {
        std::vector<std::vector<float>> samples = anchor_embeddings;
        if (samples.size() < 2) {
            add_warning(diag, "PCA requires at least 2 samples. Falling back not permitted by default.");
            return GraniteStatus::ErrArg;
        }
        out_vector = pca_first_component(samples, cfg.pca_iterations);
        if (out_vector.empty()) {
            add_warning(diag, "PCA power-iteration failed; returned empty vector.");
            return GraniteStatus::ErrBackend;
        }
        if (cfg.normalize_output && !normalize_vector(out_vector)) {
            add_warning(diag, "PCA normalization failed (zero norm). Returning raw vector.");
        }
        if (diag) {
            diag->method_used = "PCAFirstComponent";
        }
        add_note(diag, "PCA uses anchor embeddings as sample set (no external samples provided).");
        return GraniteStatus::Ok;
    }

    out_vector = mean_pool(anchor_embeddings);
    if (out_vector.empty()) {
        add_warning(diag, "Static centroid failed (dimension mismatch).");
        return GraniteStatus::ErrArg;
    }
    if (cfg.normalize_output && !normalize_vector(out_vector)) {
        add_warning(diag, "Static centroid normalization failed (zero norm). Returning raw vector.");
    }
    if (diag) {
        diag->method_used = "StaticCentroid";
    }
    return GraniteStatus::Ok;
}

} // namespace cairos
