def out_prefix(y: mut buffer[f32]):
  for i in range(1, len(y)):
    y[i] = y[i] + y[i - 1]
