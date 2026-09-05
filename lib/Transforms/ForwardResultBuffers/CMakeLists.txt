add_ondrix_library(OndrixForwardResultBuffers
  ForwardResultBuffers.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRVectorDialect
)
