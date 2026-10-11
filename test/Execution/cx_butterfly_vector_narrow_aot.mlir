// RUN: cat %S/cx_butterfly_q15_aot.mlir %s | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DTEST_VECTOR_CONSTANT_TWIDDLES %S/Inputs/cx_butterfly_q15_aot.c %t.o -o %t
// RUN: %t

func.func @bounded_vector_twiddles(%a: memref<4xi32>, %b: memref<4xi32>, %out0: memref<4xi32>, %out1: memref<4xi32>) attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : index
  %va = vector.load %a[%zero] : memref<4xi32>, vector<4xi32>
  %vb = vector.load %b[%zero] : memref<4xi32>, vector<4xi32>
  %tw = arith.constant dense<[-1518445950, -2147450881, 2147450880, -2147483648]> : vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %va, %vb, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  vector.store %0, %out0[%zero] : memref<4xi32>, vector<4xi32>
  vector.store %1, %out1[%zero] : memref<4xi32>, vector<4xi32>
  return
}

// One unbounded lane keeps the wide product even when every other lane qualifies.
func.func @unbounded_vector_twiddles(%a: memref<4xi32>, %b: memref<4xi32>, %out0: memref<4xi32>, %out1: memref<4xi32>) attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : index
  %va = vector.load %a[%zero] : memref<4xi32>, vector<4xi32>
  %vb = vector.load %b[%zero] : memref<4xi32>, vector<4xi32>
  %tw = arith.constant dense<[-1518445950, -2147450881, 2147450880, -2147450880]> : vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %va, %vb, %tw {
    layout = #ondsp.cx_layout<packed_i16_imag_hi_real_lo>,
    numeric = #ondsp.fixed<signed, storage = i16, frac = 15>,
    product = #ondsp.product<full>,
    product_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 15, rounding = nearest_even, overflow = saturate, saturate_to = i16>,
    output_scale = #ondsp.scale<pre_shift_left = 0, post_shift_right = 1, rounding = nearest_even, overflow = saturate, saturate_to = i16>
  } : (vector<4xi32>, vector<4xi32>, vector<4xi32>) -> (vector<4xi32>, vector<4xi32>)
  vector.store %0, %out0[%zero] : memref<4xi32>, vector<4xi32>
  vector.store %1, %out1[%zero] : memref<4xi32>, vector<4xi32>
  return
}
