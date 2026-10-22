def cv_tail(x: buffer[f32], k: i32, j: i64) -> f32:
  s = 0.0
  for i in range(convert(k, to=index), len(x) - convert(j, to=index)):
    s = s + x[i]
  return s
