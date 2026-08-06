// RUN: ondrix-opt %s --convert-ortumcore-to-ondsp-emulation --convert-ondsp-fixed-to-scalar --convert-scf-to-cf --finalize-memref-to-llvm --convert-arith-to-llvm --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts > %t.mlir
// RUN: ondrix-translate %t.mlir --mlir-to-llvmir > %t.ll
// RUN: llc -relocation-model=pic -filetype=obj %t.ll -o %t.o
// RUN: cc %S/Inputs/ortumcore_fft8_aot.c %t.o -o %t
// RUN: %t

// In-place radix-2 DIT FFT-8 composed ONLY from closed capabilities: the
// reorder as a bitrev_add walk (target index in the walk value's top three
// bits), each butterfly complex multiply as a two-step accumulator chain
// with the floor readout. Every stage halves, so inputs within complex
// magnitude 32000 can never saturate a store, accumulator, or readout.
func.func @ortumcore_fft8(%re: memref<8xi16>, %im: memref<8xi16>,
                          %twre: memref<4xi16>, %twim: memref<4xi16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c1_i32 = arith.constant 1 : i32
  %c29_i32 = arith.constant 29 : i32
  // rev32(1): the +1 counter step expressed in the reversed domain.
  %rev_unit = arith.constant -2147483648 : i32
  %addr0 = arith.constant 0 : i32

  %end = scf.for %k = %c0 to %c8 step %c1 iter_args(%addr = %addr0) -> (i32) {
    %j32 = arith.shrui %addr, %c29_i32 : i32
    %k32 = arith.index_cast %k : index to i32
    %later = arith.cmpi sgt, %j32, %k32 : i32
    scf.if %later {
      %j = arith.index_cast %j32 : i32 to index
      %rk = memref.load %re[%k] : memref<8xi16>
      %rj = memref.load %re[%j] : memref<8xi16>
      memref.store %rj, %re[%k] : memref<8xi16>
      memref.store %rk, %re[%j] : memref<8xi16>
      %ik = memref.load %im[%k] : memref<8xi16>
      %ij = memref.load %im[%j] : memref<8xi16>
      memref.store %ij, %im[%k] : memref<8xi16>
      memref.store %ik, %im[%j] : memref<8xi16>
    }
    %next = ortumcore.bitrev_add %addr, %rev_unit : (i32, i32) -> i32
    scf.yield %next : i32
  }

  scf.for %s = %c0 to %c3 step %c1 {
    %half = arith.shli %c1, %s : index
    %span = arith.shli %half, %c1 : index
    %stride = arith.shrui %c4, %s : index
    scf.for %base = %c0 to %c8 step %span {
      scf.for %j = %c0 to %half step %c1 {
        %widx = arith.muli %j, %stride : index
        %wr = memref.load %twre[%widx] : memref<4xi16>
        %wi = memref.load %twim[%widx] : memref<4xi16>
        %i0 = arith.addi %base, %j : index
        %i1 = arith.addi %i0, %half : index
        %xr = memref.load %re[%i0] : memref<8xi16>
        %xi = memref.load %im[%i0] : memref<8xi16>
        %yr = memref.load %re[%i1] : memref<8xi16>
        %yi = memref.load %im[%i1] : memref<8xi16>
        %zr = ortumcore.acc_init : !ortumcore.acc
        %pr = ortumcore.mac_add %zr, %wr, %yr : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
        %qr = ortumcore.mac_sub %pr, %wi, %yi : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
        %tre = ortumcore.acc_out %qr {shift = 15 : i64} : (!ortumcore.acc) -> i32
        %zi = ortumcore.acc_init : !ortumcore.acc
        %pi = ortumcore.mac_add %zi, %wr, %yi : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
        %qi = ortumcore.mac_add %pi, %wi, %yr : (!ortumcore.acc, i16, i16) -> !ortumcore.acc
        %tim = ortumcore.acc_out %qi {shift = 15 : i64} : (!ortumcore.acc) -> i32
        %xr32 = arith.extsi %xr : i16 to i32
        %xi32 = arith.extsi %xi : i16 to i32
        %sr = arith.addi %xr32, %tre : i32
        %si = arith.addi %xi32, %tim : i32
        %dr = arith.subi %xr32, %tre : i32
        %di = arith.subi %xi32, %tim : i32
        %or0 = arith.shrsi %sr, %c1_i32 : i32
        %oi0 = arith.shrsi %si, %c1_i32 : i32
        %or1 = arith.shrsi %dr, %c1_i32 : i32
        %oi1 = arith.shrsi %di, %c1_i32 : i32
        %or0n = arith.trunci %or0 : i32 to i16
        %oi0n = arith.trunci %oi0 : i32 to i16
        %or1n = arith.trunci %or1 : i32 to i16
        %oi1n = arith.trunci %oi1 : i32 to i16
        memref.store %or0n, %re[%i0] : memref<8xi16>
        memref.store %oi0n, %im[%i0] : memref<8xi16>
        memref.store %or1n, %re[%i1] : memref<8xi16>
        memref.store %oi1n, %im[%i1] : memref<8xi16>
      }
    }
  }
  return
}
