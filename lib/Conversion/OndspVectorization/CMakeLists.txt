add_ondrix_library(OndspVectorization
  OndspDecimateOutputBatching.cpp
  OndspFpFilterOutputBatching.cpp
  OndspVectorization.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMathDialect
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
