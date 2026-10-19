def basic_f32_tensor(x: tensor[f32,4], y: tensor[f32,4]) -> tensor[f32,4]:
  scale = 5e-1
  t = x * scale + y
  return -(t - 1.0) / 2.0
