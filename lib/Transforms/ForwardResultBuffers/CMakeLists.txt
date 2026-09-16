add_ondrix_library(OndrixForwardResultBuffers
  ForwardResultBuffers.cpp
  ReadOnlyArguments.cpp
  StaticResultsToOutParams.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRVectorDialect
)
