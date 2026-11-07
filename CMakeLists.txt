cmake_minimum_required(VERSION 3.14)
project(TokenSIMD C CXX)

add_compile_definitions(TSIMD_SRC_DIR="${CMAKE_SOURCE_DIR}")

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes -Wno-error)

find_package(Threads REQUIRED)

set(TSIMD_VERSION 1.0.1)
set(TSIMD_VERSION_MAJOR 1)

add_library(tokensim STATIC
  src/allocator.c
  src/trie.c
  src/input.c
  src/unicode.c
  src/unicode_tables.c
  src/json.c
  src/byte_unicode.c
  src/model_internal.c
  src/model_gpt2.c
  src/model_tokjson.c
  src/pretokenizer/pretokenizer.c
  src/pretokenizer/gpt2.c
  src/pretokenizer/cl100k.c
  src/pretokenizer/o200k.c
  src/pretokenizer/o200k_masks.c
  src/pretokenizer/cl100k_masks.c
  src/pretokenizer/nemo_masks.c
  src/pretokenizer/kimi.c
  src/pretokenizer/kimi_masks.c
  src/pretokenizer/gpt2_masks.c
  src/pretokenizer/deepseek_masks.c
  src/pretokenizer/deepseek.c
  src/metaspace.c
  src/nfc.c
  src/nfc_tables.c
  src/model_spm.c
  src/pretokenizer/simd_pretok.c
  src/pretokenizer/safe_split.c
  src/vocabulary.c
  src/bpe.c
  src/tokenizer.c
  src/sys.c
  src/spm_tie_rule.c
  src/parallel.c
  src/serve_x86.cpp
  src/simd/dispatch.c
  src/simd/blocks_scalar.c
  src/simd/blocks_sse2.c
  src/simd/blocks_avx2.c
  src/simd/blocks_avx512.c
)
set(TSIMD_X86_TARGET OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64)|(i[3-6]86)|(x86)")
  set(TSIMD_X86_TARGET ON)
elseif(APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
  set(TSIMD_X86_TARGET ON)
endif()
if(NOT TSIMD_X86_TARGET)
  message(FATAL_ERROR
    "TokenSIMD 1.x ships x86-64 kernels only (SSE2/AVX2/AVX-512); \
the pretokenizer spanners and the serve path have no non-x86 port. \
Build on x86-64, or wait for a release that declares scalar kernels.")
endif()

set_source_files_properties(src/pretokenizer/o200k_masks.c PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(src/pretokenizer/cl100k_masks.c PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(src/pretokenizer/nemo_masks.c PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(src/pretokenizer/deepseek_masks.c PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(src/pretokenizer/gpt2_masks.c PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(src/pretokenizer/kimi_masks.c PROPERTIES COMPILE_OPTIONS "-mavx2")

set_source_files_properties(src/serve_x86.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
if(TSIMD_X86_TARGET)
  set_source_files_properties(src/nfc.c PROPERTIES COMPILE_OPTIONS "-mpopcnt")
endif()

target_include_directories(tokensim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(tokensim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/simd
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pretokenizer)
target_link_libraries(tokensim PUBLIC Threads::Threads)

if(TSIMD_X86_TARGET)
  set_source_files_properties(src/simd/blocks_sse2.c PROPERTIES COMPILE_OPTIONS "-msse2")
  set_source_files_properties(src/simd/blocks_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2")
  set_source_files_properties(src/simd/blocks_avx512.c PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl")

  set_source_files_properties(src/bpe.c PROPERTIES COMPILE_OPTIONS "-msse4.2")
endif()

set_target_properties(tokensim PROPERTIES
  VERSION ${TSIMD_VERSION}
  SOVERSION ${TSIMD_VERSION_MAJOR}
  POSITION_INDEPENDENT_CODE ON)   # consumable by shared objects (Python)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  string(APPEND CMAKE_C_FLAGS " -Wno-cast-function-type")
endif()

set(TSIMD_PGO "" CACHE STRING "PGO stage: generate | use | empty")
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  if(TSIMD_PGO STREQUAL "generate")
    string(APPEND CMAKE_C_FLAGS " -fprofile-generate=${CMAKE_BINARY_DIR}/pgo-data")
    string(APPEND CMAKE_CXX_FLAGS " -fprofile-generate=${CMAKE_BINARY_DIR}/pgo-data")
  elseif(TSIMD_PGO STREQUAL "use")
    string(APPEND CMAKE_C_FLAGS " -fprofile-use=${CMAKE_BINARY_DIR}/pgo-data -Wno-missing-profile")
    string(APPEND CMAKE_CXX_FLAGS " -fprofile-use=${CMAKE_BINARY_DIR}/pgo-data -Wno-missing-profile")
  endif()
endif()
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
if(Python3_Development.Module_FOUND)
  Python3_add_library(tsimd_python MODULE WITH_SOABI bindings/python/tsimd_module.c)
  target_link_libraries(tsimd_python PRIVATE tokensim)
  set_target_properties(tsimd_python PROPERTIES OUTPUT_NAME tokensim)
  message(STATUS "Python bindings: enabled (Python ${Python3_VERSION})")
else()
  message(STATUS "Python bindings: disabled (Python3 Development.Module not found)")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS tokensim
        EXPORT tokensim-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(TARGET tsimd_python)

  set(TSIMD_PYTHON_INSTALL_DIR
      "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
  if(DEFINED SKBUILD_PLATLIB_DIR)
    install(TARGETS tsimd_python LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR})
    install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/tsimd.py
            DESTINATION ${SKBUILD_PLATLIB_DIR})
  else()
    install(TARGETS tsimd_python LIBRARY DESTINATION ${TSIMD_PYTHON_INSTALL_DIR})
    install(FILES ${CMAKE_SOURCE_DIR}/bindings/python/tsimd.py
            DESTINATION ${TSIMD_PYTHON_INSTALL_DIR})
  endif()
endif()
install(FILES
  include/tokenizer.h include/vocabulary.h include/pretokenizer.h
  include/bpe.h include/batch.h include/input.h include/allocator.h
  include/simd.h include/unicode.h include/tokensimd_version.h
  include/tokensimd_sys.h include/spm_tie_rule.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tokensim)
configure_file(cmake/tokensim.pc.in tokensim.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/tokensim.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
configure_package_config_file(
  cmake/tokensim-config.cmake.in cmake/tokensim-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tokensim)
write_basic_package_version_file(
  cmake/tokensim-config-version.cmake
  VERSION ${TSIMD_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT tokensim-targets NAMESPACE tokensim::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tokensim)
install(FILES
  ${CMAKE_BINARY_DIR}/cmake/tokensim-config.cmake
  ${CMAKE_BINARY_DIR}/cmake/tokensim-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tokensim)

add_executable(tsimd-tokenize tools/tokenize_file.c)
target_link_libraries(tsimd-tokenize PRIVATE tokensim)

add_executable(bench_encode tools/bench_encode.c)
target_link_libraries(bench_encode PRIVATE tokensim)

add_executable(bench_phases tools/bench_phases.c)
target_link_libraries(bench_phases PRIVATE tokensim)

add_executable(spm_probe tools/spm_probe.c)
target_link_libraries(spm_probe PRIVATE tokensim)

add_executable(encode_probe tools/encode_probe.c)
target_link_libraries(encode_probe PRIVATE tokensim)

add_executable(uni_boundary_probe scripts/uni_boundary_probe.c)
target_link_libraries(uni_boundary_probe PRIVATE tokensim)

add_executable(wholechain_bench scripts/wholechain_bench.c)
target_link_libraries(wholechain_bench PRIVATE tokensim)

add_executable(uni_cutclass_probe scripts/uni_cutclass_probe.c)
target_link_libraries(uni_cutclass_probe PRIVATE tokensim)

add_executable(leak_probe scripts/leak_probe.c)
target_link_libraries(leak_probe PRIVATE tokensim)

add_executable(ids_parity scripts/ids_parity.c)
target_link_libraries(ids_parity PRIVATE tokensim)

add_executable(deepseek_span_probe scripts/deepseek_span_probe.c)
target_link_libraries(deepseek_span_probe PRIVATE tokensim)

add_executable(span_probe scripts/span_probe.c)
target_link_libraries(span_probe PRIVATE tokensim)

add_executable(kimi_span_diff scripts/kimi_span_diff.c)
target_link_libraries(kimi_span_diff PRIVATE tokensim)

add_executable(corpus_hash scripts/corpus_hash.c)
target_link_libraries(corpus_hash PRIVATE tokensim)

add_executable(chunk_bisect scripts/chunk_bisect.c)
target_link_libraries(chunk_bisect PRIVATE tokensim)

add_executable(nemo_spin_probe scripts/nemo_spin_probe.c)
target_link_libraries(nemo_spin_probe PRIVATE tokensim)

add_executable(uni_par_probe scripts/uni_par_probe.c)
target_link_libraries(uni_par_probe PRIVATE tokensim)

add_executable(gemma_window_probe scripts/gemma_window_probe.c)
target_link_libraries(gemma_window_probe PRIVATE tokensim)

add_executable(gemma_unit_probe scripts/gemma_unit_probe.c)
target_link_libraries(gemma_unit_probe PRIVATE tokensim)

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  foreach(t test_added_scan test_avx512_probe test_bcls_tables test_bmp_props test_bpe test_bpe_short test_cache_state test_cl100k_masks test_cls_table test_deepseek_attach test_deepseek_masks test_fill_tail_guard test_fuzz test_gpt2_masks test_gpt2d_spanner test_hi_algebra test_hugepage_tables test_insert_slot test_json test_model_formats test_nemo_masks test_nfc_seg test_nfkc test_nul_keys test_par_split test_parallel test_parallel_into test_pcache_grow test_pcache_race test_pcache_serve test_pool_cache_bits test_pretokenizer test_reconstruct_order test_scheme_guard test_simd test_spm_normalize test_spm_trie test_tokenizer test_unicode test_worker_caps)
    add_executable(${t} tests/${t}.c tests/util.c)
    target_link_libraries(${t} PRIVATE tokensim)
    add_test(NAME ${t} COMMAND ${t} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
  endforeach()
  if(EXISTS ${CMAKE_SOURCE_DIR}/models/reference
      AND EXISTS ${CMAKE_SOURCE_DIR}/testdata/owt.txt)
    foreach(t test_o200k_masks test_batch_fused test_batch_nfc test_batch_scale test_bpe_heap_oracle test_bpe_path_agreement test_bpe_seam test_byte_ids_fallback test_cache_stress test_decode test_dual_cursor test_encode_into test_gemma_added_par test_gemma_metaspace test_giant_word_cliff test_ignore_merges test_kimi test_kimi_masks test_kimi_par_race test_mask_stats_gate test_merge_id_probe test_nemotron test_nfc test_nfc_reuse test_parallel_nfc test_pcache_upgrade test_pin_fault test_qwen3_nfc_speed test_shared_fallback test_shared_publish test_short_skip test_spm_cache test_spm_integration test_spm_par_degenerate test_spm_tie_rule test_spm_tiebreak test_spm_word_seam test_sys test_unigram test_unigram_chunk test_unigram_memo_corpus test_unigram_par test_vocabulary test_wholepiece_sim)
      add_executable(${t} tests/${t}.c tests/util.c)
      target_link_libraries(${t} PRIVATE tokensim)
      add_test(NAME ${t} COMMAND ${t} ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
    endforeach()
  endif()

  if(EXISTS ${CMAKE_SOURCE_DIR}/models/reference
      AND EXISTS ${CMAKE_SOURCE_DIR}/testdata/owt.txt)
    add_executable(test_unigram_memo tests/test_unigram_memo.c tests/util.c)
    target_link_libraries(test_unigram_memo PRIVATE tokensim)
    add_test(NAME unigram_memo_memo COMMAND test_unigram_memo memo)
    add_test(NAME unigram_memo_whole COMMAND test_unigram_memo whole)
    add_executable(test_o200k_marks tests/test_o200k_marks.c)
    target_link_libraries(test_o200k_marks PRIVATE tokensim)
    add_test(NAME o200k_marks COMMAND test_o200k_marks)

    add_executable(test_qwen25_serve tests/test_qwen25_serve.c)
    target_link_libraries(test_qwen25_serve PRIVATE tokensim)
    add_test(NAME qwen25_serve COMMAND test_qwen25_serve 2)
    set_tests_properties(qwen25_serve PROPERTIES TIMEOUT 300)

    add_executable(test_unigram_memo_det tests/test_unigram_memo_det.c)
    target_link_libraries(test_unigram_memo_det PRIVATE tokensim)
    add_test(NAME unigram_memo_det COMMAND test_unigram_memo_det 4)
    set_tests_properties(unigram_memo_det PROPERTIES TIMEOUT 300)

    add_executable(test_serve_routes tests/test_serve_routes.c tests/util.c)
    target_link_libraries(test_serve_routes PRIVATE tokensim)
    add_test(NAME serve_routes_o200k COMMAND test_serve_routes --scheme o200k)
    add_test(NAME serve_routes_cl100k COMMAND test_serve_routes --scheme cl100k)
    add_executable(test_serve_batched tests/test_serve_batched.c tests/util.c)
    target_link_libraries(test_serve_batched PRIVATE tokensim)
    add_test(NAME serve_batched_batched COMMAND test_serve_batched batched)
    add_test(NAME serve_batched_perspan COMMAND test_serve_batched perspan)

    add_executable(test_dispatch tests/test_dispatch.c)
    target_link_libraries(test_dispatch PRIVATE tokensim)
    add_test(NAME dispatch_force_avx512_falls_back COMMAND test_dispatch)
    set_tests_properties(dispatch_force_avx512_falls_back PROPERTIES
      ENVIRONMENT "TSIMD_SIMD_FORCE=avx512;EXPECT_NO_AVX512=1")
    add_test(NAME dispatch_scalar_fill COMMAND test_dispatch)
    set_tests_properties(dispatch_scalar_fill PROPERTIES
      ENVIRONMENT "TSIMD_SIMD_FORCE=scalar;EXPECT_MASK=0")
    add_test(NAME dispatch_mask_fill COMMAND test_dispatch)
    set_tests_properties(dispatch_mask_fill PROPERTIES
      ENVIRONMENT "TSIMD_SIMD_FORCE=avx2;EXPECT_MASK=1")
    add_test(NAME dispatch_hash_crc32 COMMAND test_dispatch)
    set_tests_properties(dispatch_hash_crc32 PROPERTIES
      ENVIRONMENT "TSIMD_SIMD_FORCE=avx2;TSIMD_HASH_CRC32=1;EXPECT_MASK=1")
  endif()
endif()

if(BUILD_TESTING)
add_executable(bench_pretokenizer benchmarks/bench_pretokenizer.c)
target_link_libraries(bench_pretokenizer PRIVATE tokensim)

add_executable(bench_parallel benchmarks/bench_parallel.c)
target_link_libraries(bench_parallel PRIVATE tokensim)

include(CheckCCompilerFlag)
check_c_compiler_flag("-fsanitize=fuzzer" TSIMD_HAS_LIBFUZZER)
if(TSIMD_ENABLE_LIBFUZZER AND TSIMD_HAS_LIBFUZZER)
  add_executable(fuzz_tokenizer tests/fuzz/fuzz_tokenizer.c tests/util.c)
  target_include_directories(fuzz_tokenizer PRIVATE tests)
  target_compile_options(fuzz_tokenizer PRIVATE -fsanitize=fuzzer)
  target_link_options(fuzz_tokenizer PRIVATE -fsanitize=fuzzer)
  target_compile_definitions(fuzz_tokenizer PRIVATE TSIMD_LIBFUZZER_DRIVER)
  add_test(NAME fuzz_tokenizer_libfuzzer
           COMMAND fuzz_tokenizer -runs=5000 -max_len=8192)
else()
  add_executable(fuzz_tokenizer tests/fuzz/fuzz_tokenizer.c tests/util.c)
  target_include_directories(fuzz_tokenizer PRIVATE tests)
  target_link_libraries(fuzz_tokenizer PRIVATE tokensim)
  add_test(NAME fuzz_tokenizer COMMAND fuzz_tokenizer 2000)
endif()

add_executable(bench_pretok_mt tools/bench_pretok_mt.c)
target_link_libraries(bench_pretok_mt PRIVATE tokensim)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  add_executable(fuzz_pretokenize tests/fuzz_pretokenize.c)
  target_compile_options(fuzz_pretokenize PRIVATE -fsanitize=fuzzer)
  target_link_options(fuzz_pretokenize PRIVATE -fsanitize=fuzzer)
  target_link_libraries(fuzz_pretokenize PRIVATE tokensim)

  add_executable(fuzz_all_models tests/fuzz_all_models.c)
  target_compile_options(fuzz_all_models PRIVATE -fsanitize=fuzzer)
  target_link_options(fuzz_all_models PRIVATE -fsanitize=fuzzer)
  target_link_libraries(fuzz_all_models PRIVATE tokensim)

  add_executable(fuzz_encode_vs_oracle tests/fuzz_encode_vs_oracle.c)
  target_compile_options(fuzz_encode_vs_oracle PRIVATE -fsanitize=fuzzer)
  target_link_options(fuzz_encode_vs_oracle PRIVATE -fsanitize=fuzzer)
  target_link_libraries(fuzz_encode_vs_oracle PRIVATE tokensim)
  add_executable(fuzz_encode_vs_oracle_cl100k
                 tests/fuzz_encode_vs_oracle_cl100k.c)
  target_compile_options(fuzz_encode_vs_oracle_cl100k PRIVATE
                         -fsanitize=fuzzer)
  target_link_options(fuzz_encode_vs_oracle_cl100k PRIVATE
                      -fsanitize=fuzzer)
  target_link_libraries(fuzz_encode_vs_oracle_cl100k PRIVATE tokensim)
  add_executable(fuzz_encode_vs_oracle_gpt2
                 tests/fuzz_encode_vs_oracle_gpt2.c)
  target_compile_options(fuzz_encode_vs_oracle_gpt2 PRIVATE
                         -fsanitize=fuzzer)
  target_link_options(fuzz_encode_vs_oracle_gpt2 PRIVATE
                      -fsanitize=fuzzer)
  target_link_libraries(fuzz_encode_vs_oracle_gpt2 PRIVATE tokensim)

  add_executable(fuzz_decode_roundtrip tests/fuzz_decode_roundtrip.c)
  target_compile_options(fuzz_decode_roundtrip PRIVATE -fsanitize=fuzzer)
  target_link_options(fuzz_decode_roundtrip PRIVATE -fsanitize=fuzzer)
  target_link_libraries(fuzz_decode_roundtrip PRIVATE tokensim)
endif()
endif()

set(CPACK_SOURCE_GENERATOR "TXZ")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "tokensim-${TSIMD_VERSION}-src")
set(CPACK_SOURCE_IGNORE_FILES
    "/build[^/]*/"
    "/testdata/"
    "/models/"
    "/docs/"
    "/crash-[^/]*"
    "/oom-[^/]*"
    "/timeout-[^/]*"
    "/\\.zcode/"
    "/\\.claude/"
    "/\\.git"
    "__pycache__"
    "/attic/"
    "/bindings/rust/target/"
    "/vendor/"
    "/\.pytest_cache/"
    "/dist/"
    "/\.github/"
    "/\.zcode/"
    "perf\.data"
    "\.log$"
    "\.so$")
include(CPack)
