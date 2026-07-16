add_ondrix_library(OndrixToOndsp
  OndrixToOndsp.cpp

  DEPENDS
  MLIROndrixPassesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  OndrixIR
  OndspIR
)
