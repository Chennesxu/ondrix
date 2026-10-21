def br_ratio(a: i32, b: i32) -> i32:
  return (a / b if b != 0 else 0) * 4 + (1 if a > 0 else -1 if a < 0 else 0)
