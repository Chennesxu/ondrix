add_ondrix_library(OndrixUnfuseSharedFpProducts
  UnfuseSharedFpProducts.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  OndspIR
)
