def invalid_window_return(signal: tensor[q15,16]) -> tensor[q15,9]:
  return hamming(taps=9)
