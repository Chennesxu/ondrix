// RUN: split-file %s %t
// RUN: ondrix-compile %t/mac.ox | FileCheck %s --check-prefix=MAC
// RUN: ondrix-compile %t/leak.ox | FileCheck %s --check-prefix=LEAK
// RUN: ondrix-compile %t/q31.ox | FileCheck %s --check-prefix=Q31
// RUN: not ondrix-compile %t/width.ox 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: not ondrix-compile %t/width31.ox 2>&1 | FileCheck %s --check-prefix=WIDTH31
// RUN: not ondrix-compile %t/update.ox 2>&1 | FileCheck %s --check-prefix=UPDATE
// RUN: not ondrix-compile %t/format.ox 2>&1 | FileCheck %s --check-prefix=FORMAT
// RUN: not ondrix-compile %t/import.ox 2>&1 | FileCheck %s --check-prefix=IMPORT
// RUN: not ondrix-compile %t/factor.ox 2>&1 | FileCheck %s --check-prefix=FACTOR
// RUN: not ondrix-compile %t/first.ox 2>&1 | FileCheck %s --check-prefix=FIRST
// RUN: not ondrix-compile %t/target.ox 2>&1 | FileCheck %s --check-prefix=TARGET
// RUN: not ondrix-compile %t/operator.ox 2>&1 | FileCheck %s --check-prefix=OPERATOR
// RUN: not ondrix-compile %t/result.ox 2>&1 | FileCheck %s --check-prefix=RESULT
// RUN: not ondrix-compile %t/reassign.ox 2>&1 | FileCheck %s --check-prefix=REASSIGN
// RUN: not ondrix-compile %t/literal.ox 2>&1 | FileCheck %s --check-prefix=LITERAL
// RUN: not ondrix-compile %t/legacy.ox 2>&1 | FileCheck %s --check-prefix=LEGACY
// MAC: %[[ZERO:.*]] = ondsp.acc_zero : <storage = i40, frac = 30, signed, update_overflow = saturate>
// MAC: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (!ondsp.acc<storage = i40, frac = 30, signed, update_overflow = saturate>)
// MAC: ondsp.mac %[[ACC]], %{{.*}}, %{{.*}} {numeric = #ondsp.fixed<signed, storage = i16, frac = 15>, product = #ondsp.product<full>}
// MAC: ondsp.acc_export %{{.*}} {dst = #ondsp.fixed<signed, storage = i16, frac = 15>, overflow = #ondsp.overflow<saturate>, rounding = #ondsp.rounding<nearest_ties_positive>}
// LEAK: scf.for {{.*}} iter_args(%[[Y:.*]] = %{{.*}}) -> (i16)
// LEAK: %[[IMPORTED:.*]] = ondsp.acc_import %[[Y]] {src = #ondsp.fixed<signed, storage = i16, frac = 15>} : (i16) -> !ondsp.acc<storage = i40, frac = 30, signed, update_overflow = wrap>
// LEAK: %[[SUM:.*]] = ondsp.mac %[[IMPORTED]], %arg1, %{{.*}}
// LEAK: %[[LEFT:.*]] = ondsp.mac_sub %[[SUM]], %arg1, %[[Y]]
// LEAK: ondsp.acc_export %[[LEFT]] {{.*}}rounding = #ondsp.rounding<nearest_even>
// Q31: ondsp.acc_zero : <storage = i64, frac = 62, signed, update_overflow = saturate>
// Q31: ondsp.mac {{.*}} {numeric = #ondsp.fixed<signed, storage = i32, frac = 31>, product = #ondsp.product<full>}
// Q31: ondsp.acc_export {{.*}} {dst = #ondsp.fixed<signed, storage = i32, frac = 31>, overflow = #ondsp.overflow<wrap>, rounding = #ondsp.rounding<toward_zero>}
// WIDTH: error: a q15 accumulator has exact width 40
// WIDTH31: error: a q31 accumulator has exact width 64
// UPDATE: error: unsupported update overflow mode 'clamp'
// FORMAT: error: an accumulator holds q15 or q31 products
// IMPORT: error: import takes a q15 or q31 scalar
// FACTOR: error: mac and mac_sub multiply two scalars of the accumulator's format
// FIRST: error: mac, mac_sub and export take an accumulator first
// TARGET: error: export reads an accumulator in its own format
// OPERATOR: error: an accumulator is advanced by mac or mac_sub and read by export
// RESULT: error: declared result type does not match the expression
// REASSIGN: error: a reassignment must keep the local's type
// LITERAL: error: integer literal is out of range
// LEGACY: error: accumulators are written in ordinary function bodies
//--- mac.ox
def mac40(x: buffer[q15], h: buffer[q15]) -> q15:
  acc = zero(accumulator=exact[40, saturate], of=q15)
  for i in range(len(x)):
    acc = mac(acc, x[i], h[i])
  return export(acc, to=q15)

//--- leak.ox
def leak(x: buffer[q15], a: q15) -> q15:
  y = 0
  for i in range(len(x)):
    acc = mac(import(y, accumulator=exact[40, wrap]), a, x[i])
    y = export(mac_sub(acc, a, y), to=q15, rounding=nearest_even)
  return y

//--- q31.ox
def q31mac(x: buffer[q31], h: buffer[q31]) -> q31:
  acc = zero(accumulator=exact[64, saturate], of=q31)
  for i in range(len(x)):
    acc = mac(acc, x[i], h[i])
  return export(acc, to=q31, rounding=toward_zero, overflow=wrap)

//--- width.ox
def f(x: buffer[q15]) -> q15:
  return export(zero(accumulator=exact[32, saturate], of=q15), to=q15)

//--- width31.ox
def f(x: buffer[q31]) -> q31:
  return export(zero(accumulator=exact[40, saturate], of=q31), to=q31)

//--- update.ox
def f(x: q15) -> q15:
  return export(mac(zero(accumulator=exact[40, clamp], of=q15), x, x), to=q15)

//--- format.ox
def f(x: buffer[f32]) -> f32:
  return export(zero(accumulator=exact[40, saturate], of=f32), to=f32)

//--- import.ox
def f(x: tensor[q15, 4]) -> q15:
  return export(import(x, accumulator=exact[40, wrap]), to=q15)

//--- factor.ox
def f(x: buffer[q15], w: buffer[q31]) -> q15:
  return export(mac(zero(accumulator=exact[40, saturate], of=q15), x[0], w[0]), to=q15)

//--- first.ox
def f(x: buffer[q15]) -> q15:
  return export(mac(x[0], x[0], x[0]), to=q15)

//--- target.ox
def f(x: buffer[q15]) -> q31:
  return export(zero(accumulator=exact[40, saturate], of=q15), to=q31)

//--- operator.ox
def f(x: buffer[q15]) -> q15:
  acc = zero(accumulator=exact[40, saturate], of=q15)
  return export(acc + x[0], to=q15)

//--- result.ox
def f(x: q15) -> q15:
  return mac(zero(accumulator=exact[40, saturate], of=q15), x, x)

//--- reassign.ox
def f(x: buffer[q15]) -> q15:
  acc = zero(accumulator=exact[40, saturate], of=q15)
  for i in range(len(x)):
    acc = mac(zero(accumulator=exact[40, wrap], of=q15), x[i], x[i])
  return export(acc, to=q15)

//--- literal.ox
def f(x: buffer[q15]) -> q15:
  return export(mac(zero(accumulator=exact[40, wrap], of=q15), x[0], 40000), to=q15)

//--- legacy.ox
def f(x: buffer[complex_q15, 8]) -> complex_q15:
  return export(zero(accumulator=exact[40, wrap], of=q15), to=q15)
