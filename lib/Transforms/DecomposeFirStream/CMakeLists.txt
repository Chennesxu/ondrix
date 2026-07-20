add_ondrix_library(OndrixDecomposeFirStream
  DecomposeFirStream.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTransformUtils
  OndrixFirStreamRuntimeShape
  OndrixIR
)
