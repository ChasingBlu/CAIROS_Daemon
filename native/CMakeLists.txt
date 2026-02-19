cmake_minimum_required(VERSION 3.20)
project(cairos_daemon_native LANGUAGES CXX)

option(CAIROS_WITH_ONNX "Enable ONNX Runtime backend" ON)
option(CAIROS_WITH_SECURE_LOGGER "Enable SecureLogger integration (private)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cairos_granite
    src/granite_embedder.cpp
    src/identity_anchor.cpp
    src/recp_metrics.cpp
)

add_executable(granite_embedder_cli src/granite_embedder_cli.cpp)
add_executable(granite_hidden_state_cli src/granite_hidden_state_cli.cpp)
add_executable(identity_anchor_cli src/identity_anchor_cli.cpp)
add_executable(recp_metrics_cli src/recp_metrics_cli.cpp)
add_executable(granite_tokenizer_cli src/granite_tokenizer_cli.cpp)
add_executable(coords_from_embeddings_cli src/coords_from_embeddings_cli.cpp)

foreach(tgt cairos_granite granite_embedder_cli granite_hidden_state_cli identity_anchor_cli recp_metrics_cli coords_from_embeddings_cli granite_tokenizer_cli)
    target_include_directories(${tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endforeach()

if(CAIROS_WITH_ONNX)
    target_compile_definitions(cairos_granite PRIVATE CAIROS_WITH_ONNX)
    target_compile_definitions(granite_embedder_cli PRIVATE CAIROS_WITH_ONNX)
    target_compile_definitions(granite_hidden_state_cli PRIVATE CAIROS_WITH_ONNX)
    target_compile_definitions(identity_anchor_cli PRIVATE CAIROS_WITH_ONNX)
    target_compile_definitions(recp_metrics_cli PRIVATE CAIROS_WITH_ONNX)

    if(NOT ONNXRUNTIME_DIR)
        message(FATAL_ERROR "Set ONNXRUNTIME_DIR to the ONNX Runtime root (containing include/ and lib/).")
    endif()
    if(NOT SENTENCEPIECE_DIR)
        message(FATAL_ERROR "Set SENTENCEPIECE_DIR to the SentencePiece root (containing include/ and lib/).")
    endif()

    target_include_directories(cairos_granite PRIVATE ${ONNXRUNTIME_DIR}/include ${SENTENCEPIECE_DIR}/include)
    target_link_directories(cairos_granite PRIVATE ${ONNXRUNTIME_DIR}/lib ${SENTENCEPIECE_DIR}/lib)
    target_link_libraries(cairos_granite PRIVATE onnxruntime sentencepiece)

    target_link_libraries(granite_embedder_cli PRIVATE cairos_granite)
    target_link_libraries(granite_hidden_state_cli PRIVATE cairos_granite)
    target_link_libraries(identity_anchor_cli PRIVATE cairos_granite)
    target_link_libraries(recp_metrics_cli PRIVATE cairos_granite)
    target_link_libraries(granite_tokenizer_cli PRIVATE cairos_granite)
endif()

if(CAIROS_WITH_SECURE_LOGGER)
    target_compile_definitions(granite_hidden_state_cli PRIVATE CAIROS_WITH_SECURE_LOGGER)
    target_compile_definitions(recp_metrics_cli PRIVATE CAIROS_WITH_SECURE_LOGGER)
endif()

# coords_from_embeddings_cli is a standalone tool (no ONNX dependency).


# granite_tokenizer_cli requires ONNX + SentencePiece (tokenizer parity).


