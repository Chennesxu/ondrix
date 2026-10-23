def keep(t: tensor[f32, 4]) -> tensor[f32, 4]:
  return t

def tc_outer(x: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  y = x * 2.0
  w = y
  for i in range(n):
    z = keep(y)
    y = y * 0.5 + w
    y = y + z
  return y + w
