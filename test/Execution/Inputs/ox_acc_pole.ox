def acc_pole(x: buffer[q15]) -> q15:
  y = 0
  for i in range(len(x)):
    acc = mac(zero(accumulator=exact[40, saturate], of=q15), y, 29491)
    y = export(mac(acc, x[i], 3277), to=q15)
  return y
