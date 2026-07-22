add_ondrix_library(OndrixFrontend
  OxFrontend.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRSupport
  OndrixIR
  OndspIR
  )
