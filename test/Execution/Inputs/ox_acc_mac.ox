def acc_mac(x: buffer[q15], h: buffer[q15]) -> q15:
  acc = zero(accumulator=exact[40, saturate], of=q15)
  for i in range(len(x)):
    acc = mac(acc, x[i], h[i])
  return export(acc, to=q15)
