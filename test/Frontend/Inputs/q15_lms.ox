def q15_lms(
    x: tensor[q15,64],
    d: tensor[q15,64],
    w: tensor[q15,8])
    -> (tensor[q15,64], tensor[q15,8]):
  return lms(x, d, w, step_size=4096)
