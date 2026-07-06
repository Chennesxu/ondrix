add_mlir_dialect(OrtumCoreOps ortumcore)
add_mlir_doc(OrtumCoreDialect OrtumCoreDialect ortumcore/ -gen-dialect-doc)
add_mlir_doc(OrtumCoreOps OrtumCoreOps ortumcore/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS OrtumCoreTypes.td)
mlir_tablegen(OrtumCoreTypes.h.inc -gen-typedef-decls)
mlir_tablegen(OrtumCoreTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIROrtumCoreTypesIncGen)
