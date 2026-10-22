def cv_narrow(v: i64, wrap: bool) -> i32:
  return convert(v, to=i32, overflow=wrap) if wrap else convert(v, to=i32)
