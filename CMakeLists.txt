cmake_minimum_required(VERSION 3.16)
project(asr_decoder LANGUAGES CXX)

add_library(asr_decoder STATIC
    cpp/src/context_graph.cpp
    cpp/src/context_policy.cpp
    cpp/src/ctc_decoder.cpp
    cpp/src/decoder_state.cpp
    cpp/src/greedy_search.cpp
    cpp/src/prefix_beam_search.cpp
    cpp/src/prefix_score.cpp)
add_library(asr_decoder::asr_decoder ALIAS asr_decoder)
target_compile_features(asr_decoder PUBLIC cxx_std_17)
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
endif()
