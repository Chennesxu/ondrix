// RUN: ondrix-opt %s --merge-ondrix-gain-cascades > %t.merged.ondrix.mlir
// RUN: FileCheck %s --check-prefix=MERGED < %t.merged.ondrix.mlir
// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/gain_merge_q15_aot.c %t.o -o %t
// RUN: %t
// RUN: ondrix-opt %t.merged.ondrix.mlir --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map allow-return-allocs" --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.merged.mlir
// RUN: ondrix-translate %t.merged.mlir --mlir-to-llvmir > %t.merged.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.merged.ll -o %t.merged.o
// RUN: cc %S/Inputs/gain_merge_q15_aot.c %t.merged.o -o %t.merged
// RUN: %t.merged

// Object-level replay of the gain-merge legality story over the WHOLE
// input domain: the harness sweeps all 65536 possible i16 values through
// both compiled functions in batches of 4096 (the maximum gain extent).
//
//   * The certified cascade (halve, then negate-halve) must agree with
//     the independent two-stage reference AND with the single merged
//     -8192 gain on every input — the exhaustive compile-time certificate
//     replayed at object level. Both the unmerged and the merged pipeline
//     compile this function; they must be indistinguishable.
//   * The witness cascade (22938 then 19661) survives the merge pass
//     unchanged; the harness pins the reason: the quantized-product merge
//     13763 diverges from it on exactly 10038 inputs, the swapped cascade
//     order diverges from the merge on 11418, and the two cascade orders
//     even diverge from each other on 13556 — quantized gain cascades do
//     not commute, let alone merge.
//   * The two admissible tie rules are separate executable contracts. The
//     same `gain = 3` compiled under nearest_even and under
//     nearest_ties_positive must DISAGREE on the negative tie
//     `-16384 * 3 = -49152 = -1.5 ulp` (-2 versus -1) and agree everywhere
//     else, over the whole domain.
//   * The cascade (-16384 then -8192) is certified only under
//     nearest_ties_positive, where it merges to 4096; the object gate
//     replays that certificate the same way the nearest_even one is
//     replayed.

// MERGED-LABEL: func.func @gain_cascade_certified
// MERGED: = ondrix.gain %
// MERGED-SAME: gain = -8192
// MERGED-SAME: exhaustive_inputs = 65536
// MERGED-NOT: = ondrix.gain %

// MERGED-LABEL: func.func @gain_cascade_witness
// MERGED: = ondrix.gain %
// MERGED-SAME: gain = 22938
// MERGED: = ondrix.gain %
// MERGED-SAME: gain = 19661

// MERGED-LABEL: func.func @gain_cascade_reverse_certified
// MERGED: = ondrix.gain %
// MERGED-SAME: gain = -8192
// MERGED-SAME: inner_gain = -16384
// MERGED-SAME: outer_gain = 16384
// MERGED-SAME: rounding = "nearest_even"
// MERGED-NOT: = ondrix.gain %

// MERGED-LABEL: func.func @gain_cascade_ties_positive
// MERGED: = ondrix.gain %
// MERGED-SAME: gain = 4096
// MERGED-SAME: exhaustive_inputs = 65536
// MERGED-NOT: = ondrix.gain %

// MERGED-LABEL: func.func @gain_scale_nearest_even
// MERGED: = ondrix.gain %
// MERGED-SAME: rounding = #ondsp.rounding<nearest_even>

// MERGED-LABEL: func.func @gain_scale_ties_positive
// MERGED: = ondrix.gain %
// MERGED-SAME: rounding = #ondsp.rounding<nearest_ties_positive>

func.func @gain_cascade_certified(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %halved = ondrix.gain %input {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  %negated = ondrix.gain %halved {
    gain = -16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %negated : tensor<4096xi16>
}

func.func @gain_cascade_witness(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %first = ondrix.gain %input {
    gain = 22938 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  %second = ondrix.gain %first {
    gain = 19661 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %second : tensor<4096xi16>
}

// The nearest_even-certified pair whose ties-positive counterpart is NOT
// mergeable. Its constants are the reverse order of @gain_cascade_certified,
// and that is a different witness rather than a restatement: finite-precision
// gain cascades do not commute, so (-16384 then 16384) has to be certified
// and replayed on its own.
func.func @gain_cascade_reverse_certified(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %negated = ondrix.gain %input {
    gain = -16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  %halved = ondrix.gain %negated {
    gain = 16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %halved : tensor<4096xi16>
}

// Certified only under the ties-toward-positive rule: the nearest_even
// certificate for these same constants fails on 8192 inputs.
func.func @gain_cascade_ties_positive(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %negated = ondrix.gain %input {
    gain = -16384 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  %scaled = ondrix.gain %negated {
    gain = -8192 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %scaled : tensor<4096xi16>
}

// One constant, two declared tie rules: gain = 3 puts every raw input whose
// product is an exact half ulp on the boundary that separates the modes.
func.func @gain_scale_nearest_even(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.gain %input {
    gain = 3 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}

func.func @gain_scale_ties_positive(%input: tensor<4096xi16>) -> tensor<4096xi16>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.gain %input {
    gain = 3 : i64,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_ties_positive>
  } : (tensor<4096xi16>) -> tensor<4096xi16>
  return %result : tensor<4096xi16>
}
