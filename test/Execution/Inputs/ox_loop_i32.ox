def ranges_i32(a: i32, b: i32, c: i32, d: i32) -> i32:
  acc = 0
  for k in range(a, b):
    acc = acc * 31 + k
  for k in range(a, b, 3):
    acc = acc * 31 + k
  for k in range(b, a, -1):
    acc = acc * 31 + k
  for k in range(b, a, -3):
    acc = acc * 31 + k
  for k in range(c, d, 1000003):
    acc = acc * 31 + k
  for k in range(d, c, -999983):
    acc = acc * 31 + k
  for k in range(c, d, 2147483647):
    acc = acc * 31 + k
  for k in range(d, c, -2147483648):
    acc = acc * 31 + k
  return acc
