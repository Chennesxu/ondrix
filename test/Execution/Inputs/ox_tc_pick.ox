def tc_pick(x: tensor[f32, 4], w: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  y = x
  for i in range(n):
    c = y if y[0] > 0.0 else w
    y = c * 2.0
    y = y if y[1] > 1.0 else c
  return y
