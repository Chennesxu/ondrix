// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256 fft-loops=true" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=AVX2
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=128 fft-loops=true" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEON
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=0 fft-loops=true" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCALAR
// RUN: ondrix-opt %s --ondrix-default-pipeline="vector-bits=256" --dump-pass-pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNROLLED
// RUN: ondrix-opt %S/../Execution/cfft_q31_aot.mlir --ondrix-default-pipeline="vector-bits=256 fft-loops=true" | FileCheck %s --check-prefix=Q31
// RUN: ondrix-opt %S/../Execution/cfft64_q15_aot.mlir --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=2 fft-loops-vector-register-bits=256" | FileCheck %s --check-prefix=OVERRIDE
// RUN: ondrix-opt %S/../Execution/cfft64_q15_aot.mlir --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-width=0 fft-loops-vector-register-bits=256" | FileCheck %s --check-prefix=ZERO-AUTO
// RUN: ondrix-opt %S/../Execution/cfft64_q15_aot.mlir --convert-ondrix-to-ondsp="fft-loops fft-loops-vector-register-bits=192" | FileCheck %s --check-prefix=UNSUPPORTED --implicit-check-not=vector.transfer_read
// RUN: not ondrix-opt %s --convert-ondrix-to-ondsp="fft-loops-vector-register-bits=-1" 2>&1 | FileCheck %s --check-prefix=NEGATIVE

// One target register holds eight Q15 packed values or four Q31 packed
// values at 256 bits. Direct pass users may still specify a positive lane
// count; zero derives from declared register bits when those are present.
// AVX2: convert-ondrix-to-ondsp{{.*}}fft-loops=true fft-loops-vector-register-bits=256 fft-loops-vector-width=0
// NEON: convert-ondrix-to-ondsp{{.*}}fft-loops=true fft-loops-vector-register-bits=128 fft-loops-vector-width=0
// SCALAR: convert-ondrix-to-ondsp{{.*}}fft-loops=true fft-loops-vector-register-bits=0 fft-loops-vector-width=0
// UNROLLED: convert-ondrix-to-ondsp{{.*}}fft-loops=false fft-loops-vector-register-bits=0 fft-loops-vector-width=0
// Q31: llvm.load {{.*}} : !llvm.ptr -> vector<4xi64>
// OVERRIDE: vector.transfer_read {{.*}} : tensor<64xi32>, vector<2xi32>
// ZERO-AUTO: vector.transfer_read {{.*}} : tensor<64xi32>, vector<8xi32>
// UNSUPPORTED-LABEL: func.func @cfft64_forward_q15
// NEGATIVE: fft-loops-vector-register-bits must be nonnegative

module {}
