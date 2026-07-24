def f32_dot_fast(lhs: buffer[f32], rhs: buffer[f32]) -> f32:
  return dot(lhs, rhs, contract=fast)
