// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=NARROW
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDE
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128" | FileCheck %s --check-prefix=ORDERED
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" | FileCheck %s --check-prefix=BATCHED

// A Q31 DCT requantizes every product; at 128 bits the rounded i64 lanes lose
// to the ordered schedule, so both routes decline the site and only there.

// NARROW: vectorize-ondsp-fixed-decimate-outputs{chunk-multiple=2 max-straight-line-coefficients=0 requantized-products=false vector-width=4}
// NARROW-SAME: vectorize-ondsp-fixed-memref-reduce{chunk-multiple=4 pair-fold-squares=true requantized-products=false vector-width=4}
// WIDE: vectorize-ondsp-fixed-decimate-outputs{chunk-multiple=2 max-straight-line-coefficients=0 requantized-products=true vector-width=8}
// WIDE-SAME: vectorize-ondsp-fixed-memref-reduce{chunk-multiple=4 pair-fold-squares=true requantized-products=true vector-width=8}

// ORDERED-LABEL: llvm.func @dct8_q31
// ORDERED-NOT: vector<
// BATCHED-LABEL: llvm.func @dct8_q31
// BATCHED: vector<{{[0-9]+}}xi64>

func.func @dct8_q31(%input: tensor<8xi32>) -> tensor<8xi32>
    attributes {llvm.emit_c_interface} {
  %result = ondrix.dct %input {
    input_numeric = #ondsp.fixed<signed, storage = i32, frac = 31>,
    output_numeric = #ondsp.fixed<signed, storage = i32, frac = 27>,
    product_rounding = #ondsp.rounding<nearest_even>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<8xi32>) -> tensor<8xi32>
  return %result : tensor<8xi32>
}
