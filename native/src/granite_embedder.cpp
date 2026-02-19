#include "granite_embedder.hpp"

/*
 * GraniteEmbedder (C++ implementation)
 *
 * ONNX Runtime backend + SentencePiece tokenizer
 *
 * REPA/ISO notes:
 *  - No silent fallback. If backend, model, or tokenizer is missing, we fail hard
 *    and provide a reason in last_error().
 *  - TorchScript remains explicitly unsupported here.
 */

#include <filesystem>
#include <array>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

#ifdef CAIROS_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#include <sentencepiece_processor.h>
#endif

namespace fs = std::filesystem;

namespace cairos {

struct GraniteEmbedder::GraniteImpl {
#ifdef CAIROS_WITH_ONNX
    Ort::Env env;
    Ort::SessionOptions session_opts;
    std::unique_ptr<Ort::Session> session;
    sentencepiece::SentencePieceProcessor sp;
    std::wstring onnx_path;
    std::string spm_path;
    std::string output_name;
    bool output_is_logits;
    bool pool_cls_token;
    bool pool_mean_tokens;
    bool pooling_config_loaded;
    std::size_t max_seq_len;

    GraniteImpl()
        : env(ORT_LOGGING_LEVEL_WARNING, "granite"),
          session_opts(),
          session(nullptr),
          output_name(""),
          output_is_logits(false),
          pool_cls_token(false),
          pool_mean_tokens(true),
          pooling_config_loaded(false),
          max_seq_len(512) {
        // Keep execution deterministic and bounded.
        session_opts.SetIntraOpNumThreads(1);
        session_opts.SetInterOpNumThreads(1);
        session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
#endif
};

GraniteEmbedder::GraniteEmbedder(const GraniteConfig& cfg)
    : cfg_(cfg), loaded_(false), last_error_(""), impl_(nullptr) {
    hook_.event_cb = nullptr;
    hook_.user_ctx = nullptr;
}

GraniteEmbedder::~GraniteEmbedder() = default;

void GraniteEmbedder::set_error(const std::string& msg) {
    last_error_ = msg;
}

void GraniteEmbedder::emit_event(const char* tag, const std::string& detail) const {
    if (hook_.event_cb) {
        hook_.event_cb(tag, detail.c_str(), hook_.user_ctx);
    }
}

void GraniteEmbedder::set_hook(const GraniteHook& hook) {
    hook_ = hook;
}

#ifdef CAIROS_WITH_ONNX
static bool parse_json_bool(const std::string& src, const std::string& key, bool* out);
#endif

GraniteStatus GraniteEmbedder::load() {
    emit_event("granite.load", "attempt");

    if (cfg_.backend == GraniteBackend::None) {
        set_error("Granite backend not selected. Configure GraniteConfig::backend.");
        emit_event("granite.load", "backend none");
        loaded_ = false;
        return GraniteStatus::ErrBackend;
    }

    if (cfg_.backend == GraniteBackend::TorchScript) {
        set_error("Granite TorchScript backend not integrated.");
        emit_event("granite.load", "torchscript backend missing");
        loaded_ = false;
        return GraniteStatus::ErrBackend;
    }

#ifndef CAIROS_WITH_ONNX
    set_error("Granite ONNX Runtime backend disabled at build time.");
    emit_event("granite.load", "onnx disabled");
    loaded_ = false;
    return GraniteStatus::ErrBackend;
#else
    if (cfg_.backend != GraniteBackend::OnnxRuntime) {
        set_error("Granite backend not integrated (unknown backend).");
        emit_event("granite.load", "backend missing");
        loaded_ = false;
        return GraniteStatus::ErrBackend;
    }

    if (cfg_.model_dir.empty()) {
        set_error("GraniteConfig::model_dir is empty.");
        emit_event("granite.load", "model_dir empty");
        loaded_ = false;
        return GraniteStatus::ErrArg;
    }

    const fs::path model_root(cfg_.model_dir);
    const fs::path onnx_path = model_root / "model.onnx";
    const fs::path spm_path = model_root / "sentencepiece.bpe.model";

    if (!fs::exists(onnx_path)) {
        set_error("model.onnx not found in model_dir.");
        emit_event("granite.load", "model.onnx missing");
        loaded_ = false;
        return GraniteStatus::ErrIO;
    }

    if (!fs::exists(spm_path)) {
        set_error("sentencepiece.bpe.model not found in model_dir.");
        emit_event("granite.load", "sentencepiece model missing");
        loaded_ = false;
        return GraniteStatus::ErrIO;
    }

    try {
        impl_ = std::make_unique<GraniteImpl>();
        impl_->onnx_path = onnx_path.wstring();
        impl_->spm_path = spm_path.string();

        auto status = impl_->sp.Load(impl_->spm_path);
        if (!status.ok()) {
            set_error("SentencePiece load failed: " + status.ToString());
            emit_event("granite.load", "sentencepiece load failed");
            loaded_ = false;
            return GraniteStatus::ErrBackend;
        }

        impl_->session = std::make_unique<Ort::Session>(
            impl_->env,
            impl_->onnx_path.c_str(),
            impl_->session_opts
        );

        // Determine output name. Prefer token-level outputs (rank-3) for pooling parity.
        Ort::AllocatorWithDefaultOptions allocator;
        std::string chosen;
        bool chosen_token_output = false;
        const size_t out_count = impl_->session->GetOutputCount();

        auto pick_by_rank = [&](bool require_logits, bool require_rank3) {
            for (size_t i = 0; i < out_count; ++i) {
                auto name_alloc = impl_->session->GetOutputNameAllocated(i, allocator);
                if (!name_alloc) {
                    continue;
                }
                const std::string name = name_alloc.get();
                const bool is_logits = (name == "logits");
                if (require_logits != is_logits) {
                    continue;
                }
                auto type_info = impl_->session->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                const auto shape = tensor_info.GetShape();
                const bool is_rank3 = (shape.size() == 3);
                if (require_rank3 != is_rank3) {
                    continue;
                }
                chosen = name;
                chosen_token_output = is_rank3;
                return true;
            }
            return false;
        };

        // 1) Prefer non-logits rank-3 outputs (token-level hidden state if exposed).
        if (!pick_by_rank(false, true)) {
            // 2) Then logits rank-3 outputs (often the only token-level output in Granite ONNX).
            if (!pick_by_rank(true, true)) {
                // 3) Fall back to non-logits pooled outputs.
                if (!pick_by_rank(false, false)) {
                    // 4) Last resort: logits (pooled or otherwise).
                    pick_by_rank(true, false);
                }
            }
        }

        if (chosen.empty()) {
            chosen = "logits";
            chosen_token_output = true;
        }

        impl_->output_name = chosen;
        // output_is_logits here means "token-level output" (rank-3) for pooling logic.
        impl_->output_is_logits = chosen_token_output;
        emit_event("granite.load", ("output_name=" + impl_->output_name).c_str());

        const fs::path pooling_cfg = model_root / "1_Pooling" / "config.json";
        if (fs::exists(pooling_cfg)) {
            std::ifstream cfg_in(pooling_cfg);
            std::ostringstream cfg_buf;
            cfg_buf << cfg_in.rdbuf();
            const std::string cfg_text = cfg_buf.str();
            bool cls_val = impl_->pool_cls_token;
            bool mean_val = impl_->pool_mean_tokens;
            bool cls_ok = parse_json_bool(cfg_text, "pooling_mode_cls_token", &cls_val);
            bool mean_ok = parse_json_bool(cfg_text, "pooling_mode_mean_tokens", &mean_val);
            if (cls_ok) impl_->pool_cls_token = cls_val;
            if (mean_ok) impl_->pool_mean_tokens = mean_val;
            impl_->pooling_config_loaded = cls_ok || mean_ok;
            if (!impl_->pool_cls_token && !impl_->pool_mean_tokens) {
                impl_->pool_mean_tokens = true;
                emit_event("granite.load", "pooling config invalid; default mean");
            }
        } else {
            emit_event("granite.load", "pooling config missing; default mean");
        }

        impl_->max_seq_len = 512; // From tokenizer_config.json (model_max_length)
        loaded_ = true;
        set_error("");
        emit_event("granite.load", "ok");
        return GraniteStatus::Ok;
    } catch (const Ort::Exception& e) {
        set_error(std::string("ONNX Runtime error: ") + e.what());
        emit_event("granite.load", "onnx exception");
        loaded_ = false;
        return GraniteStatus::ErrBackend;
    } catch (const std::exception& e) {
        set_error(std::string("Granite load error: ") + e.what());
        emit_event("granite.load", "exception");
        loaded_ = false;
        return GraniteStatus::ErrBackend;
    }
#endif
}

GraniteStatus GraniteEmbedder::unload() {
    emit_event("granite.unload", "requested");
    loaded_ = false;
    impl_.reset();
    set_error("");
    return GraniteStatus::Ok;
}

bool GraniteEmbedder::is_loaded() const {
    return loaded_;
}

#ifdef CAIROS_WITH_ONNX
static bool build_token_ids(const sentencepiece::SentencePieceProcessor& sp,
                            const std::string& text,
                            std::size_t max_seq_len,
                            bool add_bos_eos,
                            std::vector<int64_t>& input_ids) {
    std::vector<int> ids;
    auto status = sp.Encode(text, &ids);
    if (!status.ok()) {
        return false;
    }

    const int bos = sp.bos_id();
    const int eos = sp.eos_id();
    const int unk = sp.unk_id();

    // HF XLM-R mapping: SentencePiece IDs must be remapped to match HF vocab ids.
    // For this model:
    //   HF <s>=0, <pad>=1, </s>=2, <unk>=3, and SP pieces map to HF id = sp_id + 1.
    // This preserves parity between C++ (SentencePiece) and HF tokenization.
    const int64_t hf_bos = 0;
    const int64_t hf_eos = 2;
    const int64_t hf_unk = 3;
    auto map_id = [&](int sp_id) -> int64_t {
        if (sp_id == bos && bos >= 0) {
            return hf_bos;
        }
        if (sp_id == eos && eos >= 0) {
            return hf_eos;
        }
        if (sp_id == unk && unk >= 0) {
            return hf_unk;
        }
        if (sp_id < 0) {
            return static_cast<int64_t>(sp_id);
        }
        return static_cast<int64_t>(sp_id + 1);
    };

    input_ids.clear();
    input_ids.reserve(ids.size() + 2);

    if (add_bos_eos && bos >= 0) {
        input_ids.push_back(map_id(bos));
    }
    for (int id : ids) {
        input_ids.push_back(map_id(id));
    }
    if (add_bos_eos && eos >= 0) {
        input_ids.push_back(map_id(eos));
    }

    if (input_ids.size() > max_seq_len) {
        input_ids.resize(max_seq_len);
        if (add_bos_eos && eos >= 0) {
            input_ids[max_seq_len - 1] = map_id(eos);
        }
    }
    return true;
}

static bool parse_json_bool(const std::string& src, const std::string& key, bool* out) {
    const std::string needle = "\"" + key + "\"";
    auto pos = src.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    auto colon = src.find(":", pos + needle.size());
    if (colon == std::string::npos) {
        return false;
    }
    auto val = src.find_first_not_of(" \t\n\r", colon + 1);
    if (val == std::string::npos) {
        return false;
    }
    if (src.compare(val, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (src.compare(val, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
}

static bool build_inputs(const sentencepiece::SentencePieceProcessor& sp,
                         const std::string& text,
                         std::size_t max_seq_len,
                         std::vector<int64_t>& input_ids,
                         std::vector<int64_t>& attention_mask) {
    if (!build_token_ids(sp, text, max_seq_len, true, input_ids)) {
        return false;
    }
    attention_mask.assign(input_ids.size(), 1);
    return true;
}

static void l2_normalize(std::vector<float>& embedding) {
    double sum_sq = 0.0;
    for (float x : embedding) {
        sum_sq += static_cast<double>(x) * static_cast<double>(x);
    }
    if (sum_sq > 0.0) {
        const float inv_norm = static_cast<float>(1.0 / std::sqrt(sum_sq));
        for (auto& x : embedding) {
            x *= inv_norm;
        }
    }
}
#endif

GraniteStatus GraniteEmbedder::encode(const std::vector<std::string>& texts,
                                      std::vector<std::vector<float>>& out_embeddings) {
    if (!loaded_) {
        set_error("GraniteEmbedder not loaded.");
        emit_event("granite.encode", "not loaded");
        return GraniteStatus::ErrState;
    }

    if (texts.empty()) {
        set_error("No texts provided.");
        emit_event("granite.encode", "empty input");
        return GraniteStatus::ErrArg;
    }

#ifndef CAIROS_WITH_ONNX
    set_error("Granite ONNX Runtime backend disabled at build time.");
    emit_event("granite.encode", "onnx disabled");
    return GraniteStatus::ErrBackend;
#else
    if (!impl_ || !impl_->session) {
        set_error("Granite ONNX backend not initialized.");
        emit_event("granite.encode", "backend missing");
        return GraniteStatus::ErrBackend;
    }

    if (cfg_.precision != GranitePrecision::FP32) {
        set_error("Only FP32 supported for ONNX backend in this build.");
        emit_event("granite.encode", "precision unsupported");
        return GraniteStatus::ErrArg;
    }

    emit_event("granite.encode", "start");
    out_embeddings.clear();
    out_embeddings.reserve(texts.size());

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {impl_->output_name.c_str()};

        for (const auto& text : texts) {
            std::vector<int64_t> input_ids;
            std::vector<int64_t> attention_mask;

            if (!build_inputs(impl_->sp, text, impl_->max_seq_len, input_ids, attention_mask)) {
                set_error("SentencePiece encoding failed.");
                emit_event("granite.encode", "sentencepiece encode failed");
                return GraniteStatus::ErrBackend;
            }

            const int64_t seq_len = static_cast<int64_t>(input_ids.size());
            const int64_t dims[2] = {1, seq_len};

            Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, input_ids.data(), input_ids.size(), dims, 2);
            Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
                mem_info, attention_mask.data(), attention_mask.size(), dims, 2);

            std::array<Ort::Value, 2> inputs = {std::move(input_ids_tensor), std::move(attention_tensor)};

            auto outputs = impl_->session->Run(Ort::RunOptions{nullptr},
                                               input_names,
                                               inputs.data(),
                                               inputs.size(),
                                               output_names,
                                               1);

            if (outputs.empty()) {
                set_error("ONNX Runtime returned no outputs.");
                emit_event("granite.encode", "no outputs");
                return GraniteStatus::ErrBackend;
            }

            auto& out = outputs[0];
            auto shape_info = out.GetTensorTypeAndShapeInfo();
            auto shape = shape_info.GetShape();

            const float* data = out.GetTensorData<float>();

            if (impl_->output_is_logits) {
                if (shape.size() != 3) {
                    set_error("Unexpected logits shape from ONNX Runtime.");
                    emit_event("granite.encode", "bad logits shape");
                    return GraniteStatus::ErrBackend;
                }
                const int64_t seq = shape[1];
                const int64_t dim = shape[2];
                if (impl_->pool_cls_token) {
                    // CLS token pooling (per 1_Pooling/config.json)
                    const float* cls_token = data; // first token
                    std::vector<float> embedding(cls_token, cls_token + dim);
                    // L2 normalize to match SentenceTransformer pipeline
                    l2_normalize(embedding);
                    out_embeddings.push_back(std::move(embedding));
                } else {
                    // Mean pooling (fallback)
                    std::vector<float> embedding(static_cast<size_t>(dim), 0.0f);
                    double count = 0.0;
                    for (int64_t t = 0; t < seq; ++t) {
                        if (t < static_cast<int64_t>(attention_mask.size()) && attention_mask[static_cast<size_t>(t)] == 0) {
                            continue;
                        }
                        const float* token = data + (t * dim);
                        for (int64_t d = 0; d < dim; ++d) {
                            embedding[static_cast<size_t>(d)] += token[d];
                        }
                        count += 1.0;
                    }
                    if (count < 1.0) {
                        count = 1.0;
                    }
                    const float inv = static_cast<float>(1.0 / count);
                    for (auto& v : embedding) {
                        v *= inv;
                    }
                    // L2 normalize to match SentenceTransformer pipeline
                    l2_normalize(embedding);
                    out_embeddings.push_back(std::move(embedding));
                }
            } else {
                if (shape.size() != 2) {
                    set_error("Unexpected pooled output shape from ONNX Runtime.");
                    emit_event("granite.encode", "bad pooled shape");
                    return GraniteStatus::ErrBackend;
                }
                const int64_t dim = shape[1];
                std::vector<float> embedding;
                embedding.reserve(static_cast<size_t>(dim));
                for (int64_t i = 0; i < dim; ++i) {
                    embedding.push_back(data[i]);
                }
                // L2 normalize to match SentenceTransformer pipeline
                l2_normalize(embedding);
                out_embeddings.push_back(std::move(embedding));
            }
        }

        set_error("");
        emit_event("granite.encode", "ok");
        return GraniteStatus::Ok;
    } catch (const Ort::Exception& e) {
        set_error(std::string("ONNX Runtime error: ") + e.what());
        emit_event("granite.encode", "onnx exception");
        return GraniteStatus::ErrBackend;
    } catch (const std::exception& e) {
        set_error(std::string("Granite encode error: ") + e.what());
        emit_event("granite.encode", "exception");
        return GraniteStatus::ErrBackend;
    }
#endif
}


GraniteStatus GraniteEmbedder::tokenize(const std::string& text,
                                        std::vector<int64_t>& token_ids,
                                        bool add_bos_eos) {
    if (!loaded_) {
        set_error("GraniteEmbedder not loaded.");
        emit_event("granite.tokenize", "not loaded");
        return GraniteStatus::ErrState;
    }

#ifndef CAIROS_WITH_ONNX
    set_error("Granite ONNX Runtime backend disabled at build time.");
    emit_event("granite.tokenize", "onnx disabled");
    return GraniteStatus::ErrBackend;
#else
    if (!impl_ || !impl_->session) {
        set_error("Granite ONNX backend not initialized.");
        emit_event("granite.tokenize", "backend missing");
        return GraniteStatus::ErrBackend;
    }

    if (text.empty()) {
        set_error("Granite tokenize: text is empty.");
        emit_event("granite.tokenize", "empty text");
        return GraniteStatus::ErrArg;
    }

    if (!build_token_ids(impl_->sp, text, impl_->max_seq_len, add_bos_eos, token_ids)) {
        set_error("SentencePiece tokenization failed.");
        emit_event("granite.tokenize", "sentencepiece encode failed");
        return GraniteStatus::ErrBackend;
    }

    set_error("");
    emit_event("granite.tokenize", "ok");
    return GraniteStatus::Ok;
#endif
}

GraniteStatus GraniteEmbedder::encode_tokens(const std::string& text,
                                             std::vector<int64_t>& token_ids,
                                             std::vector<std::vector<float>>& token_embeddings) {
    if (!loaded_) {
        set_error("GraniteEmbedder not loaded.");
        emit_event("granite.encode_tokens", "not loaded");
        return GraniteStatus::ErrState;
    }

#ifndef CAIROS_WITH_ONNX
    set_error("Granite ONNX Runtime backend disabled at build time.");
    emit_event("granite.encode_tokens", "onnx disabled");
    return GraniteStatus::ErrBackend;
#else
    if (!impl_ || !impl_->session) {
        set_error("Granite ONNX backend not initialized.");
        emit_event("granite.encode_tokens", "backend missing");
        return GraniteStatus::ErrBackend;
    }

    if (!impl_->output_is_logits) {
        set_error("Token-level embeddings require logits output (seq x dim).");
        emit_event("granite.encode_tokens", "token outputs unavailable");
        return GraniteStatus::ErrBackend;
    }

    if (cfg_.precision != GranitePrecision::FP32) {
        set_error("Only FP32 supported for ONNX backend in this build.");
        emit_event("granite.encode_tokens", "precision unsupported");
        return GraniteStatus::ErrArg;
    }

    emit_event("granite.encode_tokens", "start");

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::vector<int64_t> attention_mask;
        if (!build_inputs(impl_->sp, text, impl_->max_seq_len, token_ids, attention_mask)) {
            set_error("SentencePiece encoding failed.");
            emit_event("granite.encode_tokens", "sentencepiece encode failed");
            return GraniteStatus::ErrBackend;
        }

        const int64_t seq_len = static_cast<int64_t>(token_ids.size());
        const int64_t dims[2] = {1, seq_len};

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, token_ids.data(), token_ids.size(), dims, 2);
        Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, attention_mask.data(), attention_mask.size(), dims, 2);

        std::array<Ort::Value, 2> inputs = {std::move(input_ids_tensor), std::move(attention_tensor)};
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {impl_->output_name.c_str()};

        auto outputs = impl_->session->Run(Ort::RunOptions{nullptr},
                                           input_names,
                                           inputs.data(),
                                           inputs.size(),
                                           output_names,
                                           1);

        if (outputs.empty()) {
            set_error("ONNX Runtime returned no outputs.");
            emit_event("granite.encode_tokens", "no outputs");
            return GraniteStatus::ErrBackend;
        }

        auto& out = outputs[0];
        auto shape_info = out.GetTensorTypeAndShapeInfo();
        auto shape = shape_info.GetShape();

        if (shape.size() != 3) {
            set_error("Unexpected logits shape from ONNX Runtime.");
            emit_event("granite.encode_tokens", "bad logits shape");
            return GraniteStatus::ErrBackend;
        }

        const int64_t seq = shape[1];
        const int64_t dim = shape[2];
        const float* data = out.GetTensorData<float>();

        token_embeddings.clear();
        token_embeddings.resize(static_cast<size_t>(seq), std::vector<float>(static_cast<size_t>(dim), 0.0f));

        for (int64_t t = 0; t < seq; ++t) {
            const float* token = data + (t * dim);
            auto& dst = token_embeddings[static_cast<size_t>(t)];
            for (int64_t d = 0; d < dim; ++d) {
                dst[static_cast<size_t>(d)] = token[d];
            }
        }

        set_error("");
        emit_event("granite.encode_tokens", "ok");
        return GraniteStatus::Ok;
    } catch (const Ort::Exception& e) {
        set_error(std::string("ONNX Runtime error: ") + e.what());
        emit_event("granite.encode_tokens", "onnx exception");
        return GraniteStatus::ErrBackend;
    } catch (const std::exception& e) {
        set_error(std::string("Granite encode error: ") + e.what());
        emit_event("granite.encode_tokens", "exception");
        return GraniteStatus::ErrBackend;
    }
#endif
}

GraniteStatus GraniteEmbedder::encode_tokens_raw(const std::string& text,
                                               std::vector<int64_t>& token_ids,
                                               std::vector<std::vector<float>>& token_embeddings,
                                               std::string& output_name) {
    if (!loaded_) {
        set_error("GraniteEmbedder not loaded.");
        emit_event("granite.encode_tokens_raw", "not loaded");
        return GraniteStatus::ErrState;
    }

#ifndef CAIROS_WITH_ONNX
    set_error("Granite ONNX Runtime backend disabled at build time.");
    emit_event("granite.encode_tokens_raw", "onnx disabled");
    return GraniteStatus::ErrBackend;
#else
    if (!impl_ || !impl_->session) {
        set_error("Granite ONNX backend not initialized.");
        emit_event("granite.encode_tokens_raw", "backend missing");
        return GraniteStatus::ErrBackend;
    }

    if (cfg_.precision != GranitePrecision::FP32) {
        set_error("Only FP32 supported for ONNX backend in this build.");
        emit_event("granite.encode_tokens_raw", "precision unsupported");
        return GraniteStatus::ErrArg;
    }

    emit_event("granite.encode_tokens_raw", "start");

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::vector<int64_t> attention_mask;
        if (!build_inputs(impl_->sp, text, impl_->max_seq_len, token_ids, attention_mask)) {
            set_error("SentencePiece encoding failed.");
            emit_event("granite.encode_tokens_raw", "sentencepiece encode failed");
            return GraniteStatus::ErrBackend;
        }

        const int64_t seq_len = static_cast<int64_t>(token_ids.size());
        const int64_t dims[2] = {1, seq_len};

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, token_ids.data(), token_ids.size(), dims, 2);
        Ort::Value attention_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, attention_mask.data(), attention_mask.size(), dims, 2);

        std::array<Ort::Value, 2> inputs = {std::move(input_ids_tensor), std::move(attention_tensor)};
        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {impl_->output_name.c_str()};

        auto outputs = impl_->session->Run(Ort::RunOptions{nullptr},
                                           input_names,
                                           inputs.data(),
                                           inputs.size(),
                                           output_names,
                                           1);

        if (outputs.empty()) {
            set_error("ONNX Runtime returned no outputs.");
            emit_event("granite.encode_tokens_raw", "no outputs");
            return GraniteStatus::ErrBackend;
        }

        auto& out = outputs[0];
        auto shape_info = out.GetTensorTypeAndShapeInfo();
        auto shape = shape_info.GetShape();

        int64_t batch = 1;
        int64_t seq = 0;
        int64_t dim = 0;

        if (shape.size() == 3) {
            batch = shape[0];
            seq = shape[1];
            dim = shape[2];
            if (batch > 1) {
                emit_event("granite.encode_tokens_raw", "batch>1; using first batch");
            }
        } else if (shape.size() == 2) {
            // Some ONNX exports return pooled embeddings as [1, dim] or token embeddings as [seq, dim].
            if (shape[0] == 1 && shape[1] > 1) {
                seq = 1;
                dim = shape[1];
                emit_event("granite.encode_tokens_raw", "2d pooled output");
            } else {
                seq = shape[0];
                dim = shape[1];
                emit_event("granite.encode_tokens_raw", "2d token output");
            }
        } else {
            set_error("Unexpected token output shape from ONNX Runtime.");
            emit_event("granite.encode_tokens_raw", "bad output shape");
            return GraniteStatus::ErrBackend;
        }

        const float* data = out.GetTensorData<float>();
        if (seq <= 0 || dim <= 0) {
            set_error("Invalid token output dimensions from ONNX Runtime.");
            emit_event("granite.encode_tokens_raw", "bad output dims");
            return GraniteStatus::ErrBackend;
        }

        token_embeddings.clear();
        token_embeddings.resize(static_cast<size_t>(seq), std::vector<float>(static_cast<size_t>(dim), 0.0f));

        const float* base = data;

        for (int64_t t = 0; t < seq; ++t) {
            const float* token = base + (t * dim);
            auto& dst = token_embeddings[static_cast<size_t>(t)];
            for (int64_t d = 0; d < dim; ++d) {
                dst[static_cast<size_t>(d)] = token[d];
            }
        }

        output_name = impl_->output_name;
        set_error("");
        emit_event("granite.encode_tokens_raw", "ok");
        return GraniteStatus::Ok;
    } catch (const Ort::Exception& e) {
        set_error(std::string("ONNX Runtime error: ") + e.what());
        emit_event("granite.encode_tokens_raw", "onnx exception");
        return GraniteStatus::ErrBackend;
    } catch (const std::exception& e) {
        set_error(std::string("Granite encode error: ") + e.what());
        emit_event("granite.encode_tokens_raw", "exception");
        return GraniteStatus::ErrBackend;
    }
#endif
}

GraniteDiagnostics GraniteEmbedder::diagnostics() const {
    GraniteDiagnostics d;
    d.loaded = loaded_;
    d.model_dir = cfg_.model_dir;
    d.device = cfg_.device;
    d.precision = cfg_.precision;
    d.backend = cfg_.backend;
    d.max_batch = cfg_.max_batch;
    return d;
}

const std::string& GraniteEmbedder::last_error() const {
    return last_error_;
}

} // namespace cairos
