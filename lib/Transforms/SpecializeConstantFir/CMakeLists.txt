add_ondrix_library(OndrixSpecializeConstantFir
  SpecializeConstantFir.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTransformUtils
  OndrixIR
  OndspIR
)
