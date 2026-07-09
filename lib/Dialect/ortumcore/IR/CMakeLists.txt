add_ondrix_library(OrtumCoreIR
  OrtumCoreAttrs.cpp
  OrtumCoreDialect.cpp
  OrtumCoreOps.cpp
  OrtumCoreTypes.cpp

  DEPENDS
  MLIROrtumCoreAttrsIncGen
  MLIROrtumCoreEnumsIncGen
  MLIROrtumCoreOpsIncGen
  MLIROrtumCoreTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
