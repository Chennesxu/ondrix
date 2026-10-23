def tc_nest(x: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  y = x
  for i in range(n):
    for j in range(0, 6, 2):
      y = y * 0.5 + x
    y = y - 1.0
  return y
