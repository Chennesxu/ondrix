def basic_eager(x: i64, y: i64, enabled: bool) -> bool:
  quotient = x / y
  return enabled and quotient > 0
