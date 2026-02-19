/*
 * GraniteEmbedder (C++)
 *
 * Purpose:
 *  - Native C++ interface for IBM Granite-Embedding-278M.
 *  - ONNX Runtime backend + SentencePiece tokenizer.
 *
 * REPA/ISO notes:
 *  - No silent fallbacks. Missing backend/model returns explicit error.
 *  - Optional event hook for external logging (SecureLogger integration optional).
 */
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cairos {

enum class GraniteStatus {
    Ok = 0,
    ErrArg = 1,
    ErrIO = 2,
    ErrBackend = 3,
    ErrState = 4
};

enum class GranitePrecision {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    FP64 = 3
};

enum class GraniteBackend {
    None = 0,
    OnnxRuntime = 1,
    TorchScript = 2
};

struct GraniteHook {
    void (*event_cb)(const char* tag, const char* detail, void* user_ctx);
    void* user_ctx;
};

struct GraniteConfig {
    std::string model_dir;      // Expected: path to Granite embedding weights
    std::string device;         // "cpu" or "cuda" (future)
    GranitePrecision precision; // Desired precision (default FP32)
    GraniteBackend backend;     // None / ONNX Runtime / TorchScript
    std::size_t max_batch;      // Safe batch size
    bool offline_only;          // Enforce local-only loads
};

struct GraniteDiagnostics {
    bool loaded;
    std::string model_dir;
    std::string device;
    GranitePrecision precision;
    GraniteBackend backend;
    std::size_t max_batch;
};

class GraniteEmbedder {
public:
    explicit GraniteEmbedder(const GraniteConfig& cfg);
    ~GraniteEmbedder();

    GraniteStatus load();
    GraniteStatus unload();
    bool is_loaded() const;

    // Encode texts to embeddings. Caller provides output container.
    GraniteStatus encode(const std::vector<std::string>& texts,
                         std::vector<std::vector<float>>& out_embeddings);

    // Tokenize a single text using the model tokenizer (SentencePiece).
    // When add_bos_eos is true, BOS/EOS are injected (matching model inputs).
    GraniteStatus tokenize(const std::string& text,
                           std::vector<int64_t>& token_ids,
                           bool add_bos_eos = true);

    // Encode a single text and return token-level embeddings (seq x dim).
    // Requires token-level output (logits) from ONNX backend.
    // If not available, returns ErrBackend and sets last_error().
    GraniteStatus encode_tokens(const std::string& text,
                                std::vector<int64_t>& token_ids,
                                std::vector<std::vector<float>>& token_embeddings);

    // Diagnostics-only: return raw token-level output (seq x dim) from the ONNX output
    // name selected at load time. This bypasses the logits-only guard and is intended
    // for hidden-state/logit inspection without model modification.
    GraniteStatus encode_tokens_raw(const std::string& text,
                                    std::vector<int64_t>& token_ids,
                                    std::vector<std::vector<float>>& token_embeddings,
                                    std::string& output_name);

    // Optional hooks for CLI daemon / SecureExperimentLogger
    void set_hook(const GraniteHook& hook);

    GraniteDiagnostics diagnostics() const;
    const std::string& last_error() const;

private:
    struct GraniteImpl;

    GraniteConfig cfg_;
    bool loaded_;
    std::string last_error_;
    GraniteHook hook_{};
    std::unique_ptr<GraniteImpl> impl_;

    void set_error(const std::string& msg);
    void emit_event(const char* tag, const std::string& detail) const;
};

} // namespace cairos
