// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_filter_default_contract.ox > %t.default.mlir
// RUN: ondrix-opt %t.default.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.default.llvm.mlir
// RUN: ondrix-translate %t.default.llvm.mlir --mlir-to-llvmir > %t.default.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.default.ll -o %t.default.o
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_fir_filter_explicit_contract.ox > %t.explicit.mlir
// RUN: ondrix-opt %t.explicit.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.explicit.llvm.mlir
// RUN: ondrix-translate %t.explicit.llvm.mlir --mlir-to-llvmir > %t.explicit.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.explicit.ll -o %t.explicit.o
// RUN: cc %S/Inputs/ox_default_contract_aot.c %t.default.o %t.explicit.o -o %t
// RUN: %t
// RUN: ondrix-opt %t.default.mlir --convert-ondrix-to-ondsp --widen-ondsp-exact-accumulators --convert-ondsp-to-ortumcore --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.default.target.llvm.mlir
// RUN: ondrix-translate %t.default.target.llvm.mlir --mlir-to-llvmir > %t.default.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.default.target.ll -o %t.default.target.o
// RUN: ondrix-opt %t.explicit.mlir --convert-ondrix-to-ondsp --widen-ondsp-exact-accumulators --convert-ondsp-to-ortumcore --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.explicit.target.llvm.mlir
// RUN: ondrix-translate %t.explicit.target.llvm.mlir --mlir-to-llvmir > %t.explicit.target.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.explicit.target.ll -o %t.explicit.target.o
// RUN: cc %S/Inputs/ox_default_contract_aot.c %t.default.target.o %t.explicit.target.o -o %t.target
// RUN: %t.target

// The default contract is a claim about equivalence, not a convenience: an
// inferred i35 wrapping accumulator and the spelled i40 saturating one are
// different declarations, and this gate executes both against one exact
// reference on a corpus whose products all reach the Q15 rail. The second
// binary compiles the same two programs through the target path — exactness
// widening, the ortumcore readout composition, and generic emulation — so
// the same corpus pins the default's private realization too.
