add_ondrix_library(OndrixCEntryPoints
  CEntryPoints.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  OndspIR
)
