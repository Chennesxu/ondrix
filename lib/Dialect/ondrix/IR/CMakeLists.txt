add_ondrix_library(OndrixIR
  OndrixDialect.cpp
  OndrixOps.cpp

  DEPENDS
  MLIROndrixOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
