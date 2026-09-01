def q31_lms(x: tensor[q31,64], d: tensor[q31,64], w: tensor[q31,32]) -> (tensor[q31,64], tensor[q31,32]):
  return lms(x, d, w, step_size=268435456)
