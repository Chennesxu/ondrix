add_ondrix_library(OndrixConversionUtils
  ReductionUtils.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMemRefDialect
)
