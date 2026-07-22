set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(ondrix-compile ondrix-compile.cpp)
llvm_update_compile_flags(ondrix-compile)
target_link_libraries(ondrix-compile PRIVATE
  MLIRIR
  MLIRSupport
  OndrixFrontend
  )
mlir_check_all_link_libraries(ondrix-compile)
