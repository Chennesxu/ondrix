add_ondrix_library(OndrixEvaluateFirDesign
  EvaluateFirDesign.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  OndrixIR
)
