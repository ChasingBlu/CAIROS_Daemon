#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cairos {

struct RecpMetrics {
    double ics = 0.0;  // Identity Consistency Score (anchor-centroid when available)
    double ics_pairwise = 0.0;  // Pairwise ICS (audit)
    double ics_anchor = 0.0;  // Anchor-centroid ICS (primary when anchors exist)
    bool ics_anchor_available = false;
    double api = 0.0;  // Anchor Persistence Index
    double ecr = 0.0;  // Entropy Collapse Rate
    double ldi = 0.0;  // Loop Divergence Index
    double twv = 0.0;  // Token-to-Word Variance (TCDM)
    double srv = 0.0;  // Signal Recursion Variance
    double hmv = 0.0;  // Hook Mutation Vector
    double rds = 0.0;  // Recursive Drift Score
    double mpi = 0.0;  // Mutation Pressure Index
    double sda = 0.0;  // Softmax Drift Attribution (requires logits)
    double asv = 0.0;  // Anchor Skew Value (requires logits)
    bool sda_available = false;
    bool asv_available = false;
};

// Core vector helpers
 double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
 double drift_score(const std::vector<float>& original, const std::vector<float>& current);

// Classical metrics
 double identity_consistency_score(const std::vector<std::vector<float>>& embeddings);
double identity_consistency_anchor_centroid(const std::vector<std::vector<float>>& embeddings,
                                            const std::vector<std::vector<float>>& anchor_embeddings);
 double anchor_persistence_index(const std::vector<std::string>& turns,
                                 const std::vector<std::string>& anchors);
 double entropy_shannon(const std::string& text);
 double entropy_collapse_rate(const std::vector<std::string>& turns);
 double loop_divergence_index(const std::vector<std::vector<float>>& embeddings);
 double token_to_word_variance(const std::vector<std::string>& turns);
 double token_to_word_variance_counts(const std::vector<std::string>& turns,
                                      const std::vector<int>& token_counts);

// Reinforced metrics
 double signal_recursion_variance(const std::vector<std::vector<float>>& vectors, std::size_t k);
 double hook_mutation_vector(const std::vector<std::vector<float>>& hook_embeddings,
                             const std::vector<float>& original_embedding,
                             const std::vector<float>& token_probs);
 double mutation_pressure_index(const std::vector<double>& drifts);
 double recursive_drift_score(const std::vector<std::vector<float>>& sequence_vectors);

 RecpMetrics calculate_recp_metrics(const std::vector<std::string>& turns,
                                   const std::vector<std::vector<float>>& embeddings,
                                   const std::vector<std::string>& anchors,
                                   const std::vector<std::vector<float>>& anchor_embeddings,
                                   std::size_t srv_k = 1,
                                   const std::vector<int>* token_counts = nullptr);

} // namespace cairos


