#pragma once
/*
 * Identity Anchor Modeling (C++ / ONNX)
 *
 * Purpose:
 *  - Build a stable identity anchor vector (e_id) for RECP/DIPS/ICS pipelines.
 *  - Provide multiple methods with explicit caveats and no silent fallbacks.
 *
 * REPA/ISO notes:
 *  - All failures return explicit errors via GraniteStatus.
 *  - Any fallback must be opt-in (allow_fallback_prompt_mean = true).
 */

#include "granite_embedder.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace cairos {

enum class AnchorMethod {
    StaticCentroid = 0,
    WeightedCentroid = 1,
    ContextualTokenPooling = 2,
    ContextualPromptMean = 3,
    PCAFirstComponent = 4
};

struct IdentityAnchorConfig {
    AnchorMethod method = AnchorMethod::StaticCentroid;
    std::string prompt_template = "This is about {anchors}.";
    bool add_bos_eos = true;
    bool normalize_output = true;
    bool allow_fallback_prompt_mean = false; // REPA: no silent fallback by default.
    int pca_iterations = 24;
};

struct IdentityAnchorDiagnostics {
    std::vector<std::string> warnings;
    std::vector<std::string> notes;
    std::size_t matched_token_count = 0;
    std::size_t anchor_token_count = 0;
    bool used_fallback = false;
    std::string method_used;
};

GraniteStatus build_identity_anchor(GraniteEmbedder& embedder,
                                    const std::vector<std::string>& anchors,
                                    const std::vector<float>* weights,
                                    const IdentityAnchorConfig& cfg,
                                    std::vector<float>& out_vector,
                                    IdentityAnchorDiagnostics* diag);

} // namespace cairos
