def it_div(x: tensor[i64, 8], d: tensor[i64, 8]) -> tensor[i64, 8]:
  return x / d + x / 7
