get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  OndrixSupport
  )

add_llvm_executable(ondrix-opt ondrix-opt.cpp)
llvm_update_compile_flags(ondrix-opt)
target_link_libraries(ondrix-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(ondrix-opt)
