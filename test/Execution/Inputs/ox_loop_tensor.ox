def tensor_sum(x: tensor[f32,8]) -> f32:
  s = 0.0
  for i in range(0, 8, 3):
    s = s + x[i]
  for i in range(2, len(x)):
    s = s - x[i]
  return s
