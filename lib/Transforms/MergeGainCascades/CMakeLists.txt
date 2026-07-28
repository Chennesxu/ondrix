add_ondrix_library(OndrixMergeGainCascades
  MergeGainCascades.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  OndrixIR
)
