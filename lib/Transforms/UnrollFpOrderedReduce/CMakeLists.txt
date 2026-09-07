add_ondrix_library(OndrixUnrollFpOrderedReduce
  UnrollFpOrderedReduce.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  OndspIR
)
