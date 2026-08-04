add_ondrix_library(OndrixFuseElementwiseChains
  FuseElementwiseChains.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  OndrixIR
)
