def basic_f32_compare(x: f32, y: f32, mode: i32) -> bool:
  return (mode == 0 and x == y) or (mode == 1 and x != y) or (mode == 2 and x < y) or (mode == 3 and x <= y) or (mode == 4 and x > y) or (mode == 5 and x >= y)
