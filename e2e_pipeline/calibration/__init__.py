"""Making the numbers mean what they say.

`calibration` maps a model's probability onto observed frequencies (Platt, ECE);
`calibrate` prices a risk budget by dual ascent instead of choosing a weight.
Both exist because a threshold applied to an uncalibrated score is a threshold
in model units, not real ones.
"""
