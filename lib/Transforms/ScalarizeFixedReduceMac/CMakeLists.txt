add_ondrix_library(OndrixScalarizeFixedReduceMac
  ScalarizeFixedReduceMac.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  OndrixConversionUtils
  OndspIR
)
