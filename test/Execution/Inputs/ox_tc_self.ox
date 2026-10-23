def tc_self(x: tensor[i32, 8], n: i32) -> tensor[i32, 8]:
  y = x
  for i in range(n):
    y = y + x
    y = y * 3
  return y
