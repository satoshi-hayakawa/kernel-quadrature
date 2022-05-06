import rchq_peri_sobolev as rsob

seed = 1019663834
np_seed = 316976565
times = 50
rsob.experiments(dim=1, smooth=1, times=times, seed=seed, np_seed=np_seed)
rsob.experiments(dim=1, smooth=3, times=times, seed=seed, np_seed=np_seed)
#rsob.experiments(dim=2, smooth=1, times=times, seed=seed, np_seed=np_seed)
#rsob.experiments(dim=3, smooth=3, times=times, seed=seed, np_seed=np_seed)
