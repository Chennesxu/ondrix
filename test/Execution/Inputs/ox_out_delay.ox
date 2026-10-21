def out_delay(d: mut buffer[q15], h: buffer[q15], x: q15) -> q15:
  for i in range(len(d) - 1):
    d[len(d) - 1 - i] = d[len(d) - 2 - i]
  d[0] = x
  acc = zero(accumulator=exact[40, saturate], of=q15)
  for k in range(len(d)):
    acc = mac(acc, d[k], h[k])
  return export(acc, to=q15)
