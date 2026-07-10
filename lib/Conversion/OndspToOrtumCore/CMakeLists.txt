add_ondrix_library(OndspToOrtumCore
  OndspToOrtumCore.cpp

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  OndspIR
  OrtumCoreIR
)
