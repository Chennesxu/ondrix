def invalid_nlms_q31(x: tensor[q31,64], d: tensor[q31,64], w: tensor[q31,8]) -> (tensor[q31,64], tensor[q31,8]):
  return nlms(x, d, w, step_size=8192, epsilon=16)
