def tc_read(x: tensor[i32, 8], n: i32) -> i64:
  y = x
  total = 0
  for k in range(n):
    w = y[2:6]
    for i in range(len(w)):
      total = total + convert(w[i], to=i64)
    y = convert(convert(y, to=f32) * 0.75, to=i32) + 1
  return total + convert(y[7], to=i64)
