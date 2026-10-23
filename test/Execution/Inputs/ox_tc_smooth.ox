def tc_smooth(x: tensor[f32, 16], n: i32) -> tensor[f32, 16]:
  y = x
  for i in range(n):
    y = y * 0.5 + x
  return y
