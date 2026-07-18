add_ondrix_library(OndrixConversionUtils
  ConversionLegality.cpp
  FixedPointDomainUtils.cpp
  ReductionUtils.cpp
  ValueTypeConversions.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransformUtils
  OndspIR
)

add_ondrix_library(OndrixFixedPointVectorUtils
  FixedPointVectorUtils.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRVectorDialect
  OndrixConversionUtils
  OndspIR
)
