add_ondrix_library(OndrixFixedPointSemantics
  FixedPointSemantics.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  LLVMSupport
  )

add_ondrix_library(OndrixFirStreamRuntimeShape
  FirStreamRuntimeShape.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  OndrixIR
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
  MLIRTranslateLib
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
  OndrixEvaluateFirDesign
  OndrixForwardInsertExtract
  OndrixForwardResultBuffers
  OndrixApplyLlvmArgumentAttributes
  OndrixCEntryPoints
  OndrixScalarizeFixedReduceMac
  OndrixUnrollFixedMacLoops
  OndrixUnrollFpOrderedReduce
  OndrixWidenExactAccumulators
  OndrixFuseGainIntoFir
  OndrixFuseElementwiseChains
  OndrixMergeGainCascades
  OndrixSpecializeConstantFir
  OndrixTileFirFilter
  OrtumCoreIR
  )
