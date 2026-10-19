def f32_arithmetic(x: tensor[f32,4], y: tensor[f32,4]) -> tensor[f32,4]:
  t = x * 0.5 + y
  return -t / 2.0
