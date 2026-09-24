add_ondrix_library(OndrixNarrowViterbiMetrics
  NarrowOndspViterbiMetrics.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  OndspIR
)
