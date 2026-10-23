def tc_q15(x: tensor[q15, 8], d: tensor[q15, 8], n: i32) -> tensor[q15, 8]:
  y = x
  for i in range(n):
    if y[0] > x[0]:
      y = y - d
    else:
      y = y + d
  return y
