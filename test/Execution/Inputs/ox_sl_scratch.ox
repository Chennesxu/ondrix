def sl_scratch(x: buffer[f32], y: mut buffer[f32]):
  for n in range(len(y)):
    t = zeros(f32, 4)
    for k in range(n, 4):
      t[k] = x[n]
    y[n] = t[0] + t[1] + t[2] + t[3]
