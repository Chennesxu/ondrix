def sl_trap(x: buffer[f32], a: i64, b: i64) -> f32:
  return dot(x[a:b], x[a:b], contract=off) + dot(x[2:2], x[0:0], contract=off)
