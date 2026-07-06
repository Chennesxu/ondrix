add_mlir_dialect(OndspOps ondsp)
add_mlir_doc(OndspDialect OndspDialect ondsp/ -gen-dialect-doc)
add_mlir_doc(OndspOps OndspOps ondsp/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS OndspAttrs.td)
mlir_tablegen(OndspAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(OndspAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIROndspAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS OndspTypes.td)
mlir_tablegen(OndspTypes.h.inc -gen-typedef-decls)
mlir_tablegen(OndspTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIROndspTypesIncGen)

set(LLVM_TARGET_DEFINITIONS OndspEnums.td)
mlir_tablegen(OndspEnums.h.inc -gen-enum-decls)
mlir_tablegen(OndspEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIROndspEnumsIncGen)
