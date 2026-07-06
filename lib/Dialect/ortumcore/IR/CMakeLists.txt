add_ondrix_library(OrtumCoreIR
  OrtumCoreDialect.cpp
  OrtumCoreOps.cpp
  OrtumCoreTypes.cpp

  DEPENDS
  MLIROrtumCoreOpsIncGen
  MLIROrtumCoreTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
