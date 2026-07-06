configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(ONDRIX_TEST_DEPENDS
  FileCheck
  count
  not
  ondrix-opt
  ondrix-translate
)

add_lit_testsuite(check-ondrix "Running the ondrix regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ONDRIX_TEST_DEPENDS}
)
set_target_properties(check-ondrix PROPERTIES FOLDER "Tests")

add_lit_testsuites(ONDRIX ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${ONDRIX_TEST_DEPENDS}
)

