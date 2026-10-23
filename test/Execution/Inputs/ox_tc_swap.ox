def tc_swap(x: tensor[i64, 4], w: tensor[i64, 4], n: i32) -> tensor[i64, 4]:
  y = x
  z = w
  for i in range(n):
    t = y
    y = z
    z = t * 2
  return y - z
