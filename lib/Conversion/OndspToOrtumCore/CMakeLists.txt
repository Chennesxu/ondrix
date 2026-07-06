add_ondrix_library(OndspToOrtumCore
  OndspToOrtumCore.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  OndspIR
  OrtumCoreIR
)
