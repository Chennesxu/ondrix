add_ondrix_library(OndrixFrontend
  OxFrontend.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  OndrixIR
  OndspIR
  )
