def q15_nlms(
    x: tensor[q15,64],
    d: tensor[q15,64],
    w: tensor[q15,8])
    -> (tensor[q15,64], tensor[q15,8]):
  return nlms(x, d, w, step_size=8192, epsilon=16)
