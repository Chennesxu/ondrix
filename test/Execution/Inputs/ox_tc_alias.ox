def tc_alias(x: tensor[f32, 4], w: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  y = x
  s = w
  for i in range(n):
    z = y
    v = y[1:3]
    c = y if y[0] > 0.0 else w
    y = y * 0.5 - 1.0
    s = s * 0.25 + z + y + c
    y = y + v[1]
  return s + y
