add_ondrix_library(OndspToScalar
  OndspQ15ToScalar.cpp
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
  OndrixFixedPointSemantics
  OndspIR
)
