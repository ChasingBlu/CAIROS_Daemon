#include "recp_metrics.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace cairos {

namespace {

std::string to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

std::vector<std::string> tokenize_words(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

}

 double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double av = static_cast<double>(a[i]);
        double bv = static_cast<double>(b[i]);
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    if (na <= 0.0 || nb <= 0.0) {
        return 0.0;
    }
    return dot / std::sqrt(na * nb);
}

 double drift_score(const std::vector<float>& original, const std::vector<float>& current) {
    return 1.0 - cosine_similarity(original, current);
}

 double identity_consistency_score(const std::vector<std::vector<float>>& embeddings) {
    const size_t n = embeddings.size();
    if (n < 2) {
        return 0.0;
    }
    double total_sim = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            total_sim += cosine_similarity(embeddings[i], embeddings[j]);
            ++count;
        }
    }
    if (count == 0) {
        return 0.0;
    }
    return (2.0 / (static_cast<double>(n) * (static_cast<double>(n) - 1.0))) * total_sim;
}

 double identity_consistency_anchor_centroid(const std::vector<std::vector<float>>& embeddings,
                                            const std::vector<std::vector<float>>& anchor_embeddings) {
    if (embeddings.empty() || anchor_embeddings.empty()) {
        return 0.0;
    }
    const std::size_t dim = anchor_embeddings.front().size();
    if (dim == 0) {
        return 0.0;
    }
    for (const auto& emb : anchor_embeddings) {
        if (emb.size() != dim) {
            return 0.0;
        }
    }
    std::vector<double> weights(anchor_embeddings.size(), 0.0);
    for (std::size_t i = 0; i < anchor_embeddings.size(); ++i) {
        double max_sim = -1.0;
        for (std::size_t j = 0; j < anchor_embeddings.size(); ++j) {
            if (i == j) {
                continue;
            }
            const double sim = cosine_similarity(anchor_embeddings[i], anchor_embeddings[j]);
            if (sim > max_sim) {
                max_sim = sim;
            }
        }
        if (max_sim < 0.0) {
            max_sim = 0.0;
        }
        weights[i] = 1.0 - max_sim;
    }
    double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (weight_sum <= 0.0) {
        std::fill(weights.begin(), weights.end(), 1.0);
        weight_sum = static_cast<double>(weights.size());
    }
    for (auto& w : weights) {
        w /= weight_sum;
    }
    std::vector<float> centroid(dim, 0.0f);
    for (std::size_t i = 0; i < anchor_embeddings.size(); ++i) {
        const auto& a = anchor_embeddings[i];
        const double w = weights[i];
        for (std::size_t d = 0; d < dim; ++d) {
            centroid[d] += static_cast<float>(w * static_cast<double>(a[d]));
        }
    }
    double total_sim = 0.0;
    std::size_t count = 0;
    for (const auto& emb : embeddings) {
        if (emb.size() != dim) {
            continue;
        }
        total_sim += cosine_similarity(emb, centroid);
        ++count;
    }
    if (count == 0) {
        return 0.0;
    }
    return total_sim / static_cast<double>(count);
}

 double anchor_persistence_index(const std::vector<std::string>& turns,
                                 const std::vector<std::string>& anchors) {
    const size_t k = anchors.size();
    const size_t n = turns.size();
    if (k == 0 || n == 0) {
        return 0.0;
    }
    size_t count = 0;
    std::vector<std::string> anchors_lower;
    anchors_lower.reserve(k);
    for (const auto& a : anchors) {
        anchors_lower.push_back(to_lower(a));
    }
    for (const auto& turn : turns) {
        std::string lower_turn = to_lower(turn);
        for (const auto& anchor : anchors_lower) {
            if (!anchor.empty() && lower_turn.find(anchor) != std::string::npos) {
                ++count;
            }
        }
    }
    return static_cast<double>(count) / static_cast<double>(k * n);
}

 double entropy_shannon(const std::string& text) {
    auto tokens = tokenize_words(text);
    if (tokens.empty()) {
        return 0.0;
    }
    std::unordered_map<std::string, std::size_t> counts;
    for (const auto& t : tokens) {
        ++counts[t];
    }
    const double total = static_cast<double>(tokens.size());
    double h = 0.0;
    for (const auto& kv : counts) {
        const double p = static_cast<double>(kv.second) / total;
        if (p > 0.0) {
            h += -p * std::log2(p + 1e-12);
        }
    }
    return h;
}

 double entropy_collapse_rate(const std::vector<std::string>& turns) {
    if (turns.size() < 2) {
        return 0.0;
    }
    const double h1 = entropy_shannon(turns.front());
    const double hn = entropy_shannon(turns.back());
    if (h1 <= 0.0) {
        return 0.0;
    }
    return 1.0 - (hn / h1);
}

 double loop_divergence_index(const std::vector<std::vector<float>>& embeddings) {
    const size_t n = embeddings.size();
    if (n < 2) {
        return 0.0;
    }
    double total_div = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            total_div += (1.0 - cosine_similarity(embeddings[i], embeddings[j]));
            ++count;
        }
    }
    if (count == 0) {
        return 0.0;
    }
    return (2.0 / (static_cast<double>(n) * (static_cast<double>(n) - 1.0))) * total_div;
}

 double token_to_word_variance(const std::vector<std::string>& turns) {
    std::vector<double> ratios;
    ratios.reserve(turns.size());
    for (const auto& line : turns) {
        auto words = tokenize_words(line);
        const size_t word_count = words.size();
        if (word_count > 0) {
            const double tokens = static_cast<double>(line.size());
            ratios.push_back(tokens / static_cast<double>(word_count));
        }
    }
    if (ratios.empty()) {
        return 0.0;
    }
    const double mean = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
    double var = 0.0;
    for (double v : ratios) {
        const double diff = v - mean;
        var += diff * diff;
    }
    return var / ratios.size();
}

 double token_to_word_variance_counts(const std::vector<std::string>& turns,
                                      const std::vector<int>& token_counts) {
    std::vector<double> ratios;
    ratios.reserve(turns.size());
    for (std::size_t i = 0; i < turns.size() && i < token_counts.size(); ++i) {
        auto words = tokenize_words(turns[i]);
        const size_t word_count = words.size();
        if (word_count > 0) {
            const double tokens = static_cast<double>(token_counts[i]);
            ratios.push_back(tokens / static_cast<double>(word_count));
        }
    }
    if (ratios.empty()) {
        return 0.0;
    }
    const double mean = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
    double var = 0.0;
    for (double v : ratios) {
        const double diff = v - mean;
        var += diff * diff;
    }
    return var / ratios.size();
}

 double signal_recursion_variance(const std::vector<std::vector<float>>& vectors, std::size_t k) {
    if (vectors.size() <= k || k == 0) {
        return 0.0;
    }
    std::vector<double> drift_values;
    drift_values.reserve(vectors.size() - k);
    for (size_t i = k; i < vectors.size(); ++i) {
        drift_values.push_back(1.0 - cosine_similarity(vectors[i], vectors[i - k]));
    }
    if (drift_values.empty()) {
        return 0.0;
    }
    const double mean = std::accumulate(drift_values.begin(), drift_values.end(), 0.0) / drift_values.size();
    double var = 0.0;
    for (double v : drift_values) {
        const double diff = v - mean;
        var += diff * diff;
    }
    return var / drift_values.size();
}

 double hook_mutation_vector(const std::vector<std::vector<float>>& hook_embeddings,
                             const std::vector<float>& original_embedding,
                             const std::vector<float>& token_probs) {
    const size_t t = hook_embeddings.size();
    if (t == 0) {
        return 0.0;
    }
    double total = 0.0;
    for (size_t i = 0; i < t; ++i) {
        const double sim = cosine_similarity(hook_embeddings[i], original_embedding);
        const double weight = (i < token_probs.size()) ? static_cast<double>(token_probs[i]) : 1.0;
        total += (1.0 - sim) * weight;
    }
    return total / static_cast<double>(t);
}

 double mutation_pressure_index(const std::vector<double>& drifts) {
    if (drifts.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(drifts.begin(), drifts.end(), 0.0);
    return sum / drifts.size();
}

 double recursive_drift_score(const std::vector<std::vector<float>>& sequence_vectors) {
    if (sequence_vectors.size() < 2) {
        return 0.0;
    }
    const std::vector<float>& origin = sequence_vectors.front();
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = 1; i < sequence_vectors.size(); ++i) {
        sum += drift_score(origin, sequence_vectors[i]);
        ++count;
    }
    if (count == 0) {
        return 0.0;
    }
    return sum / static_cast<double>(count);
}

 RecpMetrics calculate_recp_metrics(const std::vector<std::string>& turns,
                                   const std::vector<std::vector<float>>& embeddings,
                                   const std::vector<std::string>& anchors,
                                   const std::vector<std::vector<float>>& anchor_embeddings,
                                   std::size_t srv_k,
                                   const std::vector<int>* token_counts) {
    RecpMetrics m;
    const double pairwise = identity_consistency_score(embeddings);
    m.ics_pairwise = pairwise;
    if (!anchor_embeddings.empty()) {
        m.ics_anchor = identity_consistency_anchor_centroid(embeddings, anchor_embeddings);
        m.ics_anchor_available = true;
        // Track A alignment: anchor-centroid is primary when anchors exist.
        m.ics = m.ics_anchor;
    } else {
        m.ics = pairwise;
    }
    m.api = anchor_persistence_index(turns, anchors);
    m.ecr = entropy_collapse_rate(turns);
    m.ldi = loop_divergence_index(embeddings);
    if (token_counts && token_counts->size() == turns.size()) {
        m.twv = token_to_word_variance_counts(turns, *token_counts);
    } else {
        m.twv = token_to_word_variance(turns);
    }
    m.srv = signal_recursion_variance(embeddings, srv_k);

    // HMV and MPI/RDS derived from available vectors; requires hook embeddings for HMV.
    // Placeholder values set to zero; callers can supply hook embeddings via a different interface.
    m.hmv = 0.0;

    m.rds = recursive_drift_score(embeddings);

    std::vector<double> drifts;
    drifts.reserve(embeddings.size());
    if (!embeddings.empty()) {
        const auto& origin = embeddings.front();
        for (const auto& v : embeddings) {
            drifts.push_back(drift_score(origin, v));
        }
    }
    m.mpi = mutation_pressure_index(drifts);

    // SDA/ASV require logits/token probabilities; leave unavailable in this pipeline.
    m.sda_available = false;
    m.asv_available = false;
    m.sda = 0.0;
    m.asv = 0.0;
    return m;
}

} // namespace cairos

