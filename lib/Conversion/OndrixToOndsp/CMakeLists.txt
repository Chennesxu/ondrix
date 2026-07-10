add_ondrix_library(OndrixToOndsp
  OndrixToOndsp.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  OndrixIR
  OndspIR
)
