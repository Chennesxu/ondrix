def br_gcd(x: i64, y: i64) -> i64:
  a = x if x >= 0 else -x
  b = y if y >= 0 else -y
  while b != 0:
    if a < b:
      t = a
      a = b
      b = t
    else:
      a = a - b
  return a
