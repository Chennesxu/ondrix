configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ONDRIX_TEST_DEPENDS
  FileCheck
  count
  llc
  not
  ondrix-constant-sequence-analysis-test
  ondrix-fixed-point-prefix-range-analysis-test
  ondrix-fixed-point-semantics-test
  ondrix-ondsp-semantics-test
  ondrix-ortumcore-target-profile-test
  ondrix-opt
  ondrix-translate
  ondrix-compile
)

add_llvm_executable(ondrix-constant-sequence-analysis-test EXCLUDE_FROM_ALL
  Support/ConstantSequenceAnalysisTest.cpp
)
llvm_update_compile_flags(ondrix-constant-sequence-analysis-test)
target_link_libraries(ondrix-constant-sequence-analysis-test PRIVATE
  OndrixConstantSequenceAnalysis
)

add_llvm_executable(ondrix-fixed-point-prefix-range-analysis-test EXCLUDE_FROM_ALL
  Support/FixedPointPrefixRangeAnalysisTest.cpp
)
llvm_update_compile_flags(ondrix-fixed-point-prefix-range-analysis-test)
target_link_libraries(ondrix-fixed-point-prefix-range-analysis-test PRIVATE
  OndrixFixedPointPrefixRangeAnalysis
)

add_llvm_executable(ondrix-fixed-point-semantics-test EXCLUDE_FROM_ALL
  Support/FixedPointSemanticsTest.cpp
)
llvm_update_compile_flags(ondrix-fixed-point-semantics-test)
target_link_libraries(ondrix-fixed-point-semantics-test PRIVATE OndrixFixedPointSemantics)

add_llvm_executable(ondrix-ondsp-semantics-test EXCLUDE_FROM_ALL
  Support/OndspSemanticsTest.cpp
)
llvm_update_compile_flags(ondrix-ondsp-semantics-test)
target_link_libraries(ondrix-ondsp-semantics-test PRIVATE OndspIR)

add_llvm_executable(ondrix-ortumcore-target-profile-test EXCLUDE_FROM_ALL
  Support/OrtumCoreTargetProfileTest.cpp
)
llvm_update_compile_flags(ondrix-ortumcore-target-profile-test)
target_link_libraries(ondrix-ortumcore-target-profile-test PRIVATE OrtumCoreTargetProfile)

add_lit_testsuite(check-ondrix "Running the ondrix regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ONDRIX_TEST_DEPENDS}
)
set_target_properties(check-ondrix PROPERTIES FOLDER "Tests")

add_lit_testsuites(ONDRIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${ONDRIX_TEST_DEPENDS}
)
