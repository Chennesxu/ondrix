def ranges_i64(a: i64, b: i64, c: i64, d: i64) -> i64:
  acc = 0
  for k in range(a, b):
    acc = acc * 31 + k
  for k in range(a, b, 3):
    acc = acc * 31 + k
  for k in range(b, a, -1):
    acc = acc * 31 + k
  for k in range(b, a, -3):
    acc = acc * 31 + k
  for k in range(c, d, 1000000000000000003):
    acc = acc * 31 + k
  for k in range(d, c, -999999999999999989):
    acc = acc * 31 + k
  for k in range(c, d, 9223372036854775807):
    acc = acc * 31 + k
  for k in range(d, c, -9223372036854775808):
    acc = acc * 31 + k
  return acc
