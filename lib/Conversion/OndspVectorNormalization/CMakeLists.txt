add_ondrix_library(OndspVectorNormalization
  OndspVectorNormalization.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRVectorDialect
  OndrixConversionUtils
  OndspIR
)
