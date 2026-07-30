add_ondrix_library(OndrixFuseGainIntoFir
  FuseGainIntoFir.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  OndrixIR
)
