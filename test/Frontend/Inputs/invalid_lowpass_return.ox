def invalid_lowpass_return(input: tensor[q15,9]) -> tensor[q15,9]:
  return lowpass(taps=9, cutoff=[1,4])
