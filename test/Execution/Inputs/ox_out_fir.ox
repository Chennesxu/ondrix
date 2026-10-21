def out_fir(x: buffer[q15], h: buffer[q15], y: mut buffer[q15]):
  for n in range(len(y)):
    acc = zero(accumulator=exact[40, saturate], of=q15)
    for k in range(len(h)):
      acc = mac(acc, x[n + k], h[k])
    y[n] = export(acc, to=q15)
