def overrun(x: buffer[f32]) -> f32:
  s = 0.0
  for i in range(len(x)):
    s = s + x[i + 1]
  return s
