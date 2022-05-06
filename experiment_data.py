import measure_reduction as mred

np_seed = 847610340

mred.experiments(data_name='3Dnet', kernel='Gaussian', np_seed=np_seed)
mred.experiments(data_name='PPlant', kernel='Gaussian', np_seed=np_seed)
