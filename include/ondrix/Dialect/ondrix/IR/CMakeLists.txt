add_mlir_dialect(OndrixOps ondrix)
add_mlir_doc(OndrixDialect OndrixDialect ondrix/ -gen-dialect-doc)
add_mlir_doc(OndrixOps OndrixOps ondrix/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS OndrixAttrs.td)
mlir_tablegen(OndrixAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(OndrixAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIROndrixAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS OndrixEnums.td)
mlir_tablegen(OndrixEnums.h.inc -gen-enum-decls)
mlir_tablegen(OndrixEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIROndrixEnumsIncGen)
