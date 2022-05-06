import rchq_measure_reduction as rred

np_seed = 20922295
times = 50
rred.experiments(data_name='3Dnet', times=times,
                 kernel='Gaussian', np_seed=np_seed)
rred.experiments(data_name='PPlant', times=times,
                 kernel='Gaussian', np_seed=np_seed)
