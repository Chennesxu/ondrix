add_ondrix_library(OndspToOrtumCore
  OndspToOrtumCore.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  OndrixConversionUtils
  OndspIR
  OrtumCoreIR
  OrtumCoreTargetProfile
)
