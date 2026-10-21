def acc_q31(x: buffer[q31], h: buffer[q31]) -> q31:
  acc = zero(accumulator=exact[64, saturate], of=q31)
  for i in range(len(x)):
    acc = mac(acc, x[i], h[i])
  return export(acc, to=q31, rounding=toward_zero, overflow=wrap)
