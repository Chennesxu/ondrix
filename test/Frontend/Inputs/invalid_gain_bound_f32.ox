def f32_bad(lhs: buffer[f32,8], rhs: buffer[f32,8]) -> f32:
  return dot(lhs, rhs, gain_bound=2)
