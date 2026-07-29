def lms_step(x: tensor[q15,8], d: tensor[q15,8], w: tensor[q15,4]) -> (tensor[q15,8], tensor[q15,4]):
  return lms(x, d, w, step_size=40000)
