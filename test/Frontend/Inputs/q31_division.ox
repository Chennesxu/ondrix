def q31_division(x: tensor[q31,8]) -> tensor[q31,8]:
  return div(x, divisor=2147483647, rounding=toward_zero, overflow=wrap) + x / 1
