def q31_ratio(x: tensor[q31,16], y: tensor[q31,16]) -> tensor[q31,16]:
  return x / (abs(y) + 1) + ratio(x, shift(abs(y), amount=-3) + 1024, rounding=toward_negative, overflow=wrap)
