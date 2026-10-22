def cv_round64(v: f32, mode: i32) -> i64:
  r = 0
  if mode == 0:
    r = convert(v, to=i64, rounding=toward_zero)
  elif mode == 1:
    r = convert(v, to=i64, rounding=toward_negative)
  elif mode == 2:
    r = convert(v, to=i64, rounding=nearest_even)
  elif mode == 3:
    r = convert(v, to=i64, overflow=saturate)
  elif mode == 4:
    r = convert(2.0e19, to=i64)
  else:
    r = convert(-2.0e19, to=i64, rounding=toward_negative)
  return r
