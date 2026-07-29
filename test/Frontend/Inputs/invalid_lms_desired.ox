def lms_desired(x: tensor[q15,8], d: tensor[q15,7], w: tensor[q15,4]) -> (tensor[q15,8], tensor[q15,4]):
  return lms(x, d, w, step_size=4096)
