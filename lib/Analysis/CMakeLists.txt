add_ondrix_library(OndrixConstantSequenceAnalysis
  ConstantSequenceAnalysis.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRIR
  MLIRMemRefDialect
)

add_ondrix_library(OndrixFixedPointPrefixRangeAnalysis
  FixedPointPrefixRangeAnalysis.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRSupport
  OndrixConstantSequenceAnalysis
  OndspIR
)

add_ondrix_library(OndrixCanonicalTwiddleAnalysis
  CanonicalTwiddleAnalysis.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSupport
  OndspIR
)
