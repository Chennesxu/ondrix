def mean(x: buffer[q31]) -> f32:
  s = 0.0
  for i in range(len(x)):
    s = s + dequantize(x[i], to=f32)
  return s
