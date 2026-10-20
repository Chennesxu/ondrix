def energy(x: buffer[f32]) -> f32:
  total = 0.0
  for i in range(len(x)):
    total = total + x[i] * x[i]
  return total
