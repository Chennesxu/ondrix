def acc_leak(x: buffer[q15], a: q15) -> q15:
  y = 0
  for i in range(len(x)):
    acc = mac(import(y, accumulator=exact[40, wrap]), a, x[i])
    y = export(mac_sub(acc, a, y), to=q15, rounding=nearest_even)
  return y
