cmake_minimum_required(VERSION 3.16)
project(asr_decoder LANGUAGES CXX)

find_package(sentencepiece CONFIG QUIET)
foreach(candidate IN ITEMS
        sentencepiece::sentencepiece
        sentencepiece::sentencepiece-static
        sentencepiece
        sentencepiece-static)
    if(TARGET ${candidate})
        set(_asr_decoder_sentencepiece_target ${candidate})
        break()
    endif()
endforeach()

foreach(candidate IN ITEMS
        sentencepiece::sentencepiece_train
        sentencepiece::sentencepiece_train-static
        sentencepiece_train
        sentencepiece_train-static)
    if(TARGET ${candidate})
        set(_asr_decoder_sentencepiece_trainer_target ${candidate})
        break()
    endif()
endforeach()

if(NOT _asr_decoder_sentencepiece_target)
    include(FetchContent)
    set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
    set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        sentencepiece
        URL https://github.com/google/sentencepiece/releases/download/v0.2.1/sentencepiece-0.2.1.tar.gz
        URL_HASH SHA256=8138cec27c2f2282f4a34d9a016e3374cd40e5c6e9cb335063db66a0a3b71fad
        SOURCE_SUBDIR sentencepiece
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(sentencepiece)
    set(_asr_decoder_sentencepiece_target sentencepiece)
    set(_asr_decoder_sentencepiece_trainer_target sentencepiece_train)
endif()

# The bundled 0.2.1 source archive has an extra sentencepiece/ directory,
# while a parent project may already have populated a regular source archive.
# Older upstream CMake targets do not publish their source include directory.
if(EXISTS "${sentencepiece_SOURCE_DIR}/sentencepiece/src/sentencepiece_processor.h")
    set(_asr_decoder_sentencepiece_include_dirs
        ${sentencepiece_SOURCE_DIR}/sentencepiece/src
        ${sentencepiece_BINARY_DIR}/sentencepiece/src)
elseif(EXISTS "${sentencepiece_SOURCE_DIR}/src/sentencepiece_processor.h")
    set(_asr_decoder_sentencepiece_include_dirs
        ${sentencepiece_SOURCE_DIR}/src
        ${sentencepiece_BINARY_DIR}/src)
endif()

add_library(asr_decoder STATIC
    cpp/src/context_graph.cpp
    cpp/src/context_policy.cpp
    cpp/src/ctc_decoder.cpp
    cpp/src/decoder_state.cpp
    cpp/src/greedy_search.cpp
    cpp/src/prefix_beam_search.cpp
    cpp/src/prefix_score.cpp
    cpp/src/sentencepiece_tokenizer.cpp)
add_library(asr_decoder::asr_decoder ALIAS asr_decoder)
target_compile_features(asr_decoder PUBLIC cxx_std_17)
target_link_libraries(asr_decoder PRIVATE ${_asr_decoder_sentencepiece_target})
target_include_directories(
    asr_decoder PRIVATE ${_asr_decoder_sentencepiece_include_dirs})
target_include_directories(
    asr_decoder
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
           $<INSTALL_INTERFACE:include>)

if(MSVC)
    target_compile_options(asr_decoder PRIVATE /W4)
else()
    target_compile_options(asr_decoder PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    add_executable(asr_decoder_test cpp/tests/ctc_decoder_test.cpp)
    target_link_libraries(asr_decoder_test PRIVATE asr_decoder)
    add_test(NAME asr_decoder_test COMMAND asr_decoder_test)

    if(_asr_decoder_sentencepiece_trainer_target)
        add_executable(
            asr_decoder_sentencepiece_test
            cpp/tests/sentencepiece_tokenizer_test.cpp)
        target_link_libraries(
            asr_decoder_sentencepiece_test
            PRIVATE asr_decoder ${_asr_decoder_sentencepiece_trainer_target})
        target_include_directories(
            asr_decoder_sentencepiece_test
            PRIVATE ${_asr_decoder_sentencepiece_include_dirs})
        target_compile_definitions(
            asr_decoder_sentencepiece_test PRIVATE
            ASR_DECODER_TEST_CORPUS="${CMAKE_CURRENT_SOURCE_DIR}/cpp/tests/sentencepiece_corpus.txt"
            ASR_DECODER_TEST_MODEL_PREFIX="${CMAKE_CURRENT_BINARY_DIR}/sentencepiece_context_test")
        add_test(
            NAME asr_decoder_sentencepiece_test
            COMMAND asr_decoder_sentencepiece_test)
    endif()
endif()
