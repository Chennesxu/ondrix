add_ondrix_library(OndrixForwardResultBuffers
  ForwardResultBuffers.cpp
  StaticResultsToOutParams.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRVectorDialect
)
