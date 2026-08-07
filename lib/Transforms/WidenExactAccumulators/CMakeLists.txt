add_ondrix_library(OndrixWidenExactAccumulators
  WidenExactAccumulators.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  OndspIR
)
