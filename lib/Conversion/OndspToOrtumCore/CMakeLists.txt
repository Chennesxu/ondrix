add_ondrix_library(OndspToOrtumCore
  OndspCxButterflyToOrtumCore.cpp
  OndspLanePairsToOrtumCore.cpp
  OndspLaneSplitToOrtumCore.cpp
  OndspToOrtumCore.cpp
  OrtumCoreLoweringSupport.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFTransforms
  MLIRSupport
  MLIRTransformUtils
  MLIRVectorDialect
  OndrixConstantSequenceAnalysis
  OndrixConversionUtils
  OndrixFixedPointPrefixRangeAnalysis
  OndspIR
  OrtumCoreIR
  OrtumCoreTargetProfile
)
