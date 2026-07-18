add_ondrix_library(OndspVectorization
  OndspVectorization.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRVectorDialect
  OndrixConversionUtils
  OndrixFixedPointVectorUtils
  OndrixConstantSequenceAnalysis
  OndrixFixedPointPrefixRangeAnalysis
  OndspIR
)
