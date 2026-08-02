get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(ondrix-compile ondrix-compile.cpp)
llvm_update_compile_flags(ondrix-compile)
target_link_libraries(ondrix-compile PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRIR
  MLIRPass
  MLIRSupport
  OndrixFrontend
  OndrixPipelines
  OndrixSupport
  )
mlir_check_all_link_libraries(ondrix-compile)
