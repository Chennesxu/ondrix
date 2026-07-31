add_ondrix_library(OndrixForwardInsertExtract
  ForwardInsertExtract.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTensorDialect
)
