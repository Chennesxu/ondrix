add_ondrix_library(OndrixFrontend
  OxFrontend.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRTensorDialect
  OndrixIR
  OndspIR
  )
