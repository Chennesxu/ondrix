add_ondrix_library(OndrixConversionUtils
  ReductionUtils.cpp
  StructuralTypeConversions.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRTransformUtils
)
