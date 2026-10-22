def cv_mean(x: buffer[f32]) -> f32:
  s = 0.0
  for i in range(len(x)):
    s = s + x[i]
  return s / convert(len(x), to=f32)
