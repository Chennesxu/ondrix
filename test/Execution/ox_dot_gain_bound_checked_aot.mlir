// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot_gain_bound.ox --checked-entries --emit=llvmir > %t.checked.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot_gain_bound.ox --emit=llvmir > %t.trusted.ll
// RUN: ondrix-compile %S/../Frontend/Inputs/q15_dot_gain_bound.ox --emit=c-header > %t.h
// RUN: llc -relocation-model=pic -filetype=obj %t.checked.ll -o %t.checked.o
// RUN: llc -relocation-model=pic -filetype=obj %t.trusted.ll -o %t.trusted.o
// RUN: cc -include %t.h -DCHECKED=1 %S/Inputs/ox_dot_gain_bound_checked_aot.c %t.checked.o -o %t.checked
// RUN: cc -include %t.h -DCHECKED=0 %S/Inputs/ox_dot_gain_bound_checked_aot.c %t.trusted.o -o %t.trusted
// RUN: %t.checked
// RUN: %t.trusted

// Checked, a table at the bound's edge passes and one unit past it aborts;
// trusted, the same violation runs, since a declaration is not an enforcement.
