set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(ondrixc ondrixc.cpp)
llvm_update_compile_flags(ondrixc)
target_link_libraries(ondrixc PRIVATE
  MLIRIR
  MLIRSupport
  OndrixFrontend
  )
mlir_check_all_link_libraries(ondrixc)
