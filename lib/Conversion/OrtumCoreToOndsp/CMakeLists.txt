add_ondrix_library(OrtumCoreToOndsp
  OrtumCoreToOndsp.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTransformUtils
  OndrixConversionUtils
  OndspIR
  OrtumCoreIR
)
