add_ondrix_library(OndspVectorNormalization
  OndspPackedOperandForwarding.cpp
  OndspVectorNormalization.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRPass
  MLIRTransformUtils
  MLIRVectorDialect
  OndrixConversionUtils
  OndrixFixedPointVectorUtils
  OndspIR
)
