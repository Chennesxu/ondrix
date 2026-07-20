add_ondrix_library(OndrixFixedPointSemantics
  FixedPointSemantics.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  LLVMSupport
  )

add_ondrix_library(OndrixSupport
  InitAllDialects.cpp
  InitAllPasses.cpp
  InitAllTranslations.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIROndrixPassesIncGen
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  OndrixBufferizationInterfaces
  OndrixMemRefToSCF
  OndrixFixedPointSemantics
  OndrixIR
  OndspIR
  OrtumCoreToOndsp
  OndrixToOndsp
  OndspToScalar
  OndspVectorNormalization
  OndspVectorization
  OndspToOrtumCore
  OndrixDecomposeFirStream
  OndrixSpecializeConstantFir
  OndrixTileFirFilter
  OrtumCoreIR
  )
