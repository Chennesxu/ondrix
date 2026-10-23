def tc_mix(x: tensor[i32, 4], n: i32) -> tensor[i32, 4]:
  y = x
  for i in range(n):
    k = 0
    while k < 3:
      if y[0] > 100:
        y = y - 7
      y = y + 1
      k = k + 1
    z = y
    y = z * 2 - y
  return y
