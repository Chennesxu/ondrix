add_ondrix_library(OndrixToOndsp
  OndrixElementwiseLowering.cpp
  OndrixFirFamilyLowering.cpp
  OndrixSpectralLowering.cpp
  OndrixStatefulLowering.cpp
  OndrixToOndsp.cpp
  OndrixToOndspCommon.cpp

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
