// RUN: ondrix-opt %s --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --empty-tensor-to-alloc-tensor --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" --buffer-deallocation --expand-strided-metadata --lower-affine --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/legality_magnitude_fusion_q15_aot.c %t.o -o %t
// RUN: %t

// Legality counterexample for the candidate rewrite that fuses (or
// unfuses) a square/add/sqrt chain across explicit Q1.15 requantization
// boundaries. In exact real arithmetic, deleting the two intermediate
// requantizations does not change the value — both graphs then compute
// sqrt(re^2 + im^2) — so the rewrite is a valid real-arithmetic identity.
// Under the Ondsp contract each intermediate rounding is an observable
// boundary: the two programs below are distinct contracts and neither
// direction of the rewrite is legal. Both sides compile through their
// explicit Ondsp boundaries and are checked bit-exactly, including
// diverging witnesses such as (200,100) -> 224 vs 181 and
// (50,20) -> 54 vs 0.

// The fused contract: one exact i64 sum of squares, one rounding boundary.
func.func @magnitude_fused_q15(%bin: i32) -> i16 {
  %input = tensor.from_elements %bin : tensor<1xi32>
  %magnitudes = ondrix.cx_magnitude %input {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    rounding = #ondsp.rounding<nearest_even>
  } : (tensor<1xi32>) -> tensor<1xi16>
  %zero = arith.constant 0 : index
  %value = tensor.extract %magnitudes[%zero] : tensor<1xi16>
  return %value : i16
}

// The unfused contract: each square is requantized to Q1.15 before the
// saturating add and the square root. Every boundary is explicit.
func.func @magnitude_requantized_q15(%bin: i32) -> i16 {
  %shift = arith.constant 16 : i32
  %real16 = arith.trunci %bin : i32 to i16
  %high = arith.shrui %bin, %shift : i32
  %imaginary16 = arith.trunci %high : i32 to i16
  %real = arith.extsi %real16 : i16 to i64
  %imaginary = arith.extsi %imaginary16 : i16 to i64
  %realSquare = arith.muli %real, %real : i64
  %imaginarySquare = arith.muli %imaginary, %imaginary : i64
  %realQ15 = ondsp.round_shift %realSquare {
    scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15,
                         rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i64) -> i16
  %imaginaryQ15 = ondsp.round_shift %imaginarySquare {
    scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15,
                         rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (i64) -> i16
  %realWide = arith.extsi %realQ15 : i16 to i64
  %imaginaryWide = arith.extsi %imaginaryQ15 : i16 to i64
  %sum = arith.addi %realWide, %imaginaryWide : i64
  %fifteen = arith.constant 15 : i64
  %scaled = arith.shli %sum, %fifteen : i64
  %root = ondsp.sqrt_fixed %scaled {
    rounding = #ondsp.rounding<nearest_even>
  } : (i64) -> i16
  return %root : i16
}
