import numpy as np
import matplotlib.pyplot as plt

intermediate_data_root = '../data/'

data_all_good_op_path = intermediate_data_root + '/data_all.npy'
sort_good_ind_path = intermediate_data_root +'sort_good_ind.npy'
data_top_path = intermediate_data_root +'data_top_50.npy'

# --------------------------------------------------------------
# -- Distribution of values of each feature --------------------
# --------------------------------------------------------------

IS_LOAD_DATA_ALL = False

if IS_LOAD_DATA_ALL:
    # -- load the data of all features
    data = np.load(data_all_good_op_path)
    # -- load the sorting array
    sort_good_ind = np.load(sort_good_ind_path)
    # -- pick top 50 features
    data_top = data[:,sort_good_ind[:50]]
    # -- save reduced data array
    np.save(data_top_path,data_top)
else:
    data_top = np.load(data_top_path)


# -- plot the distribution of values
data_top = np.ma.masked_invalid(data_top)
plt.figure()
plt.title('Distribution of normalised values for top features')

for i in range(5):
    plt.hist(data_top[:,i][~data_top.mask[:,i]],bins=100,histtype='step')
    
     
# --------------------------------------------------------------
# -- Distribution of statistics of each feature --------------------
# --------------------------------------------------------------

all_classes_avg_out_path = intermediate_data_root+'/all_classes_avg.npy'
op_id_order_path = intermediate_data_root +'/op_id_order.npy'

all_classes_avg = np.load(all_classes_avg_out_path)
op_id_order = np.load(op_id_order_path)

stats_best = all_classes_avg[:,op_id_order[:50]]
stats_best = np.ma.masked_invalid(stats_best)
plt.figure()
plt.title('Distribution of Statistics scores for top features')
for i in range(5):
    plt.hist(stats_best[:,i][~stats_best.mask[:,i]],bins=10,histtype='step')

plt.show()











