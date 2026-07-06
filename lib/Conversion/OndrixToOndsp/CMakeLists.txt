add_ondrix_library(OndrixToOndsp
  OndrixToOndsp.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  OndrixIR
  OndspIR
)
