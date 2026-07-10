add_ondrix_library(OndrixSupport
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
  OndspToOrtumCore
  OrtumCoreIR
  )
