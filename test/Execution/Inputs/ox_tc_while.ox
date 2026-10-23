def tc_while(x: tensor[i64, 4], limit: i64) -> tensor[i64, 4]:
  y = x
  while y[0] < limit:
    y = y * 2 + 1
  return y
