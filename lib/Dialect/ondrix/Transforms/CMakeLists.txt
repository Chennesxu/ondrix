add_ondrix_library(OndrixBufferizationInterfaces
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  OndrixIR
  OndspIR
)
