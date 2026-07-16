add_ondrix_library(OndrixConversionUtils
  ConversionLegality.cpp
  ReductionUtils.cpp
  ValueTypeConversions.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransformUtils
)
