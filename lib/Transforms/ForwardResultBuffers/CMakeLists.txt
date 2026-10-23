add_ondrix_library(OndrixForwardResultBuffers
  ForwardResultBuffers.cpp
  HoistRegionBuffers.cpp
  ReadOnlyArguments.cpp
  StaticResultsToOutParams.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRVectorDialect
)
