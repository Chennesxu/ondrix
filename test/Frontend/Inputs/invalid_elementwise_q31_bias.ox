def invalid_elementwise_q31_bias(x: tensor[q31,32]) -> tensor[q31,32]:
  return offset(x, bias=2147483648)
