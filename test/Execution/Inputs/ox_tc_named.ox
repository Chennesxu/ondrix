def tc_named(x: tensor[f32, 4], n: i32) -> tensor[f32, 4]:
  z = x * 2.0
  y = x
  s = x
  for i in range(n):
    t = y * 2.0
    y = t
    y = y + 1.0
    s = s + t
    if s[0] > 50.0:
      y = z
  return s + y
