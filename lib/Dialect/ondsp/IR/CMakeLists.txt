add_ondrix_library(OndspIR
  OndspAttrs.cpp
  OndspDialect.cpp
  OndspOps.cpp
  OndspSemantics.cpp
  OndspTypes.cpp

  DEPENDS
  MLIROndspAttrsIncGen
  MLIROndspEnumsIncGen
  MLIROndspOpsIncGen
  MLIROndspTypesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
