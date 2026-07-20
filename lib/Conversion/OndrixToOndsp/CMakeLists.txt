add_ondrix_library(OndrixToOndsp
  OndrixToOndsp.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSupport
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
  OndrixFirStreamRuntimeShape
  OndrixIR
  OndspIR
)
