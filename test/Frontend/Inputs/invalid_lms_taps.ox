def lms_taps(x: tensor[q15,8], d: tensor[q15,8], w: tensor[q15,65]) -> (tensor[q15,8], tensor[q15,65]):
  return lms(x, d, w, step_size=4096)
