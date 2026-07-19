add_ondrix_library(OndrixIR
  OndrixAttrs.cpp
  OndrixDialect.cpp
  OndrixOps.cpp

  DEPENDS
  MLIROndrixAttrsIncGen
  MLIROndrixEnumsIncGen
  MLIROndrixOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRDestinationStyleOpInterface
  MLIRSupport
  MLIRTensorDialect
  MLIRTilingInterface
  OndspIR
)
