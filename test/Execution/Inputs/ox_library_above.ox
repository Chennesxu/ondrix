def above(x: buffer[f32], w: buffer[f32], t: f32) -> bool:
  return dot(x, w, contract=off) > t
