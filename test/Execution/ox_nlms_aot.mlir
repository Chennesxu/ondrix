// RUN: ondrix-compile %S/../Frontend/Inputs/q15_nlms.ox --emit=llvmir > %t.nlms.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_nlms.ox --emit=c-header > %t.nlms.h
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_lms.ox --emit=llvmir > %t.lms.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_lms.ox --emit=c-header > %t.lms.h
// RUN: llc -relocation-model=pic -filetype=obj %t.nlms.ll -o %t.nlms.o
// RUN: llc -relocation-model=pic -filetype=obj %t.lms.ll -o %t.lms.o
// RUN: cc -include %t.nlms.h -include %t.lms.h %S/Inputs/ox_nlms_aot.c %t.nlms.o %t.lms.o -o %t
// RUN: %t
