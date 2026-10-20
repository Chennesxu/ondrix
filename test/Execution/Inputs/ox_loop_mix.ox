def mix(x: buffer[f32], y: buffer[f32]) -> f32:
  s = 0.0
  for i in range(len(x) - 1, -1, -1):
    row = 0.0
    for j in range(1, len(y), 2):
      row = row + x[i] * y[j - 1]
    s = s * 0.5 + row
  return s
