def br_diff(x: buffer[f32], y: mut buffer[f32]):
  for i in range(len(x)):
    if i + 1 < len(x):
      y[i] = x[i + 1] - x[i]
    else:
      t = zeros(f32, 2)
      t[1] = x[i]
      y[i] = t[0] - t[1]
