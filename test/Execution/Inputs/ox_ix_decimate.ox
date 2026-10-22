def ix_decimate(x: buffer[f32], y: mut buffer[f32]):
  for i in range(len(y)):
    y[i] = x[2 * i] + x[i * 2 + 1]
