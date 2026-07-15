add_ondrix_library(OndrixSupport
  FixedPointSemantics.cpp
  InitAllDialects.cpp
  InitAllPasses.cpp
  InitAllTranslations.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  OndrixIR
  OndspIR
  OndrixToOndsp
  OndspToScalar
  OndspToOrtumCore
  OrtumCoreIR
  )
