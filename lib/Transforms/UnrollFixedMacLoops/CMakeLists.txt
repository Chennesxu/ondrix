add_ondrix_library(OndrixUnrollFixedMacLoops
  UnrollFixedMacLoops.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  OndspIR
)
