def ix_hold(x: buffer[q15], y: mut buffer[q15]):
  for i in range(len(y)):
    y[i] = x[(i - 1) / 2]
