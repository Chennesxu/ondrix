// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=NARROW
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0 scalar-register-bits=64" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDE
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0" | FileCheck %s --check-prefix=GROUPED
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0 scalar-register-bits=64" | FileCheck %s --check-prefix=DECLARED

// What a certified group buys is one accumulator-width add where the declared
// form takes one per product, so its worth is the machine's register width.
// The narrower machine is assumed, and the wider one declares itself.

// NARROW: scalarize-ondsp-certified-constant-reduce{{.*}}scalar-register-bits=32
// WIDE: scalarize-ondsp-certified-constant-reduce{{.*}}scalar-register-bits=64

// A straight-lined reduction has no per-term loop overhead for a block to
// amortize, so the wider machine leaves it to the definitional expansion.
// Both forms multiply in 32 bits; only the grouped one ADDS there, and only
// the declared one still carries the accumulator's saturation.
// GROUPED-LABEL: llvm.func @dot64
// GROUPED: llvm.add %{{.*}} : i32
// GROUPED-NOT: llvm.intr.smax
// DECLARED-LABEL: llvm.func @dot64
// DECLARED: llvm.intr.smax
// DECLARED-NOT: llvm.add %{{.*}} : i32

memref.global "private" constant @taps : memref<64xi16> = dense<3>

func.func @dot64(%x: memref<64xi16>) -> i16 {
  %c = memref.get_global @taps : memref<64xi16>
  %z = ondsp.acc_zero : !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %acc = ondsp.reduce_mac %z, %x, %c {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>, memref<64xi16>, memref<64xi16>) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>
  %out = ondsp.acc_export %acc {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, rounding = #ondsp.rounding<nearest_even>, overflow = #ondsp.overflow<saturate>} : (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>) -> i16
  return %out : i16
}
