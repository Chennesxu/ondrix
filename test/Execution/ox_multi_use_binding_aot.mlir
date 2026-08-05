// RUN: ondrix-compile --emit=contracts %S/../Frontend/Inputs/q15_multi_use_binding.ox | ondrix-opt --ondrix-default-pipeline="vector-bits=256" > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ox_multi_use_binding_aot.c %t.o -o %t
// RUN: %t

// A local read three times is the program that reads one value three times,
// not three independent evaluations: the duplicated Pure subtrees collapse
// and the executed object must still match an independent reference that
// evaluates the product once.
