import nys_mred as mred
import nys_sob as sob

np_seed = 847610340

# mred.experiments(data_name='3Dnet', kernel='Gaussian',
#                  times=10, np_seed=np_seed)
# mred.experiments(data_name='PPlant', kernel='Gaussian',
#                  times=10, np_seed=np_seed)

sob.experiments(dim=3, smooth=1, times=3, np_seed=np_seed)
