// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_q15.ox --emit=llvmir > %t.q15.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_q15.ox --emit=c-header > %t.q15.h
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_q15_even.ox --emit=llvmir > %t.even.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_q15_even.ox --emit=c-header > %t.even.h
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_q31.ox --emit=llvmir > %t.q31.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_q31.ox --emit=c-header > %t.q31.h
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_q31_zero.ox --emit=llvmir > %t.zero.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_q31_zero.ox --emit=c-header > %t.zero.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dequantize.ox --emit=llvmir > %t.dq15.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dequantize.ox --emit=c-header > %t.dq15.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_dequantize.ox --emit=llvmir > %t.dq31.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q31_dequantize.ox --emit=c-header > %t.dq31.h
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_chain.ox --emit=llvmir > %t.chain.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/f32_quantize_chain.ox --emit=c-header > %t.chain.h
// RUN: llc -relocation-model=pic -filetype=obj %t.q15.ll -o %t.q15.o
// RUN: llc -relocation-model=pic -filetype=obj %t.even.ll -o %t.even.o
// RUN: llc -relocation-model=pic -filetype=obj %t.q31.ll -o %t.q31.o
// RUN: llc -relocation-model=pic -filetype=obj %t.zero.ll -o %t.zero.o
// RUN: llc -relocation-model=pic -filetype=obj %t.dq15.ll -o %t.dq15.o
// RUN: llc -relocation-model=pic -filetype=obj %t.dq31.ll -o %t.dq31.o
// RUN: llc -relocation-model=pic -filetype=obj %t.chain.ll -o %t.chain.o
// RUN: cc -include %t.q15.h -include %t.even.h -include %t.q31.h -include %t.zero.h -include %t.dq15.h -include %t.dq31.h -include %t.chain.h %S/Inputs/ox_quantize_aot.c %t.q15.o %t.even.o %t.q31.o %t.zero.o %t.dq15.o %t.dq31.o %t.chain.o -lm -o %t
// RUN: %t
