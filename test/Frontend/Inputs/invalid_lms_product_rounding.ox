def q15_lms_product(x: tensor[q15,64], d: tensor[q15,64], w: tensor[q15,32]) -> (tensor[q15,64], tensor[q15,32]):
  return lms(x, d, w, step_size=4096, product_rounding=nearest_even)
