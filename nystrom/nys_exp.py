import nys_sob as sob

np_seed = 847610340

# Fig1
sob.experiments(dim=1, smooth=1, np_seed=np_seed)
sob.experiments(dim=2, smooth=1, np_seed=np_seed)
sob.experiments(dim=3, smooth=3, np_seed=np_seed)

# Fig2
# sob.experiments(dim=1, smooth=2, np_seed=np_seed)
# sob.experiments(dim=1, smooth=2, np_seed=np_seed, pow_int=3)