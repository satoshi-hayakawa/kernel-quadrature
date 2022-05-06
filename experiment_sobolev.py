import peri_sobolev as psob

seed = 1019663834
np_seed = 316976565

psob.experiments(dim=1, smooth=1, seed=seed, np_seed=np_seed)
psob.experiments(dim=1, smooth=3, seed=seed, np_seed=np_seed)
psob.experiments(dim=2, smooth=1, seed=seed, np_seed=np_seed)
psob.experiments(dim=3, smooth=3, seed=seed, np_seed=np_seed)
