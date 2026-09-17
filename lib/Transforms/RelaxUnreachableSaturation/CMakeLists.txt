add_ondrix_library(OndrixRelaxUnreachableSaturation
  RelaxUnreachableSaturation.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  OndspIR
)
