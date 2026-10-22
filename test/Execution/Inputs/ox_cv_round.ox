def cv_round(v: f32, mode: i32) -> i32:
  r = 0
  if mode == 0:
    r = convert(v, to=i32, rounding=toward_zero)
  elif mode == 1:
    r = convert(v, to=i32, rounding=toward_negative)
  elif mode == 2:
    r = convert(v, to=i32, rounding=nearest_even)
  elif mode == 3:
    r = convert(v, to=i32)
  elif mode == 4:
    r = convert(3.0e9, to=i32)
  else:
    r = convert(-3.0e9, to=i32, rounding=toward_negative)
  return r
