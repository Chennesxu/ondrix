add_ondrix_library(OndspToScalar
  OndspFixedToScalar.cpp
  OndspToScalar.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTransformUtils
  MLIRVectorDialect
  OndrixCanonicalTwiddleAnalysis
  OndrixConversionUtils
  OndrixFixedPointSemantics
  OndspIR
)
