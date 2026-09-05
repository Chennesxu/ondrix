add_ondrix_library(OndrixApplyLlvmArgumentAttributes
  ApplyLlvmArgumentAttributes.cpp

  DEPENDS
  MLIROndrixTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
)
