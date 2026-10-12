// RUN: cat %S/cx_butterfly_q15_aot.mlir %s | ondrix-opt --convert-ondrix-to-ondsp --convert-ondsp-fixed-to-scalar --convert-vector-to-scf --convert-vector-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc -DTEST_VECTOR_TABLE_TWIDDLES %S/Inputs/cx_butterfly_q15_aot.c %t.o -o %t
// RUN: %t

memref.global "private" constant @table_safe : memref<8xi32> = dense<[32767, 0xA57E5A82, 0x80000000, 0x80007FFF, 0x7FFF8000, 32767, 0, 32767]>
memref.global "private" constant @table_unsafe : memref<8xi32> = dense<[32767, 0xA57E5A82, 0x80000000, 0x80007FFF, 0x7FFF8000, 32767, 0, 0x80008000]>

func.func @safe_table_vector(%a: memref<4xi32>, %b: memref<4xi32>, %out0: memref<4xi32>, %out1: memref<4xi32>, %i: index) attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : index
  %va = vector.load %a[%zero] : memref<4xi32>, vector<4xi32>
  %vb = vector.load %b[%zero] : memref<4xi32>, vector<4xi32>
  %t = memref.get_global @table_safe : memref<8xi32>
  %pad = arith.constant 0 : i32
  %w = vector.transfer_read %t[%i], %pad {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %va, %vb, %w {
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

func.func @unsafe_table_vector(%a: memref<4xi32>, %b: memref<4xi32>, %out0: memref<4xi32>, %out1: memref<4xi32>, %i: index) attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : index
  %va = vector.load %a[%zero] : memref<4xi32>, vector<4xi32>
  %vb = vector.load %b[%zero] : memref<4xi32>, vector<4xi32>
  %t = memref.get_global @table_unsafe : memref<8xi32>
  %w = vector.load %t[%i] : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %va, %vb, %w {
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

func.func @masked_table_vector(%a: memref<4xi32>, %b: memref<4xi32>, %out0: memref<4xi32>, %out1: memref<4xi32>, %i: index) attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : index
  %va = vector.load %a[%zero] : memref<4xi32>, vector<4xi32>
  %vb = vector.load %b[%zero] : memref<4xi32>, vector<4xi32>
  %t = memref.get_global @table_safe : memref<8xi32>
  %pad = arith.constant 0x80008000 : i32
  %mask = arith.constant dense<[true, true, true, false]> : vector<4xi1>
  %w = vector.transfer_read %t[%i], %pad, %mask {in_bounds = [true]} : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %va, %vb, %w {
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

func.func @padded_table_vector(%a: memref<4xi32>, %b: memref<4xi32>, %out0: memref<4xi32>, %out1: memref<4xi32>, %i: index) attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : index
  %va = vector.load %a[%zero] : memref<4xi32>, vector<4xi32>
  %vb = vector.load %b[%zero] : memref<4xi32>, vector<4xi32>
  %t = memref.get_global @table_safe : memref<8xi32>
  %pad = arith.constant 0x80008000 : i32
  %w = vector.transfer_read %t[%i], %pad : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %va, %vb, %w {
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

func.func @safe_table_load_vector(%a: memref<4xi32>, %b: memref<4xi32>, %out0: memref<4xi32>, %out1: memref<4xi32>, %i: index) attributes {llvm.emit_c_interface} {
  %zero = arith.constant 0 : index
  %va = vector.load %a[%zero] : memref<4xi32>, vector<4xi32>
  %vb = vector.load %b[%zero] : memref<4xi32>, vector<4xi32>
  %t = memref.get_global @table_safe : memref<8xi32>
  %start = arith.constant 4 : index
  %w = vector.load %t[%start] : memref<8xi32>, vector<4xi32>
  %0, %1 = ondsp.cx_butterfly %va, %vb, %w {
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
