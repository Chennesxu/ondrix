def cv_widen(x: buffer[f32], k: i32) -> i64:
  high = convert(k, to=i64) * 65536
  return high + convert(len(x) - 3, to=i64) * 2 + convert(convert(len(x), to=i32), to=i64)
