// RUN: ondrix-reduction-window-analysis-test | FileCheck %s

// The quantity that decides a reduction's code shape: what consecutive
// windows share, which the straight-line form must keep in registers.

// CHECK: reduction window analysis: PASS
