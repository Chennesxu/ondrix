add_ondrix_library(OndrixBufferizationInterfaces
  BufferizableOpInterfaceImpl.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  OndrixIR
  OndspIR
)
