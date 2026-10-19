add_ondrix_library(OndrixFrontend
  OxFrontend.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSupport
  MLIRTensorDialect
  OndrixIR
  OndspIR
  )
