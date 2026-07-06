set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(ondrix-translate ondrix-translate.cpp)
llvm_update_compile_flags(ondrix-translate)
target_link_libraries(ondrix-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  OndrixSupport
  )
mlir_check_all_link_libraries(ondrix-translate)

