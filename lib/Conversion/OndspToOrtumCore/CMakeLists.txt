add_ondrix_library(OndspToOrtumCore
  OndspLanePairsToOrtumCore.cpp
  OndspToOrtumCore.cpp
  OrtumCoreLoweringSupport.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRSCFTransforms
  MLIRSupport
  MLIRTransformUtils
  MLIRVectorDialect
  OndrixConversionUtils
  OndspIR
  OrtumCoreIR
  OrtumCoreTargetProfile
)
