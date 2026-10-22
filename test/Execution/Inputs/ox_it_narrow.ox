def it_narrow(x: tensor[i64, 8]) -> tensor[i32, 8]:
  return convert(x, to=i32) - convert(x, to=i32, overflow=wrap)
