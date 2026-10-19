def scalar_predicate(x: i64, y: i64, enabled: bool) -> bool:
  return enabled and (y != 0 and x / y > 2)
