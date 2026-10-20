def blend(x: buffer[f32], w: buffer[f32], n: i32) -> f32:
  s = 0.0
  for i in range(n):
    s = s * 0.5 + dot(x, w, contract=off)
  return s
