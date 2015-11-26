import numpy as np
import matplotlib.pyplot as plt


intermediate_data_root = '../data/'
all_classes_avg_out_path = intermediate_data_root+'/all_classes_avg.npy'
op_id_good_path = intermediate_data_root +'/op_id_good.npy'


# -- Load the data
all_classes_avg = np.load(all_classes_avg_out_path)
op_id_good = np.load(op_id_good_path)
# -- Mask NaN entries
all_classes_avg_good = np.ma.masked_invalid(all_classes_avg[:,op_id_good])

print all_classes_avg_good.shape

#all_classes_avg_good_sort = np.ma.sort(all_classes_avg_good,axis=-1)
all_classes_avg_good_mean = np.ma.mean(all_classes_avg_good,axis=-1)
all_classes_avg_good_min = np.ma.min(all_classes_avg_good,axis=-1)


plt.bar(np.arange(39)+0.5,all_classes_avg_good_min/all_classes_avg_good_mean)
print all_classes_avg_good_min/all_classes_avg_good_mean
plt.ylim([-0.01,.85])
plt.xlabel('ID of problem')
plt.ylabel('min(ustat) / mean(ustat)')
plt.savefig('/home/philip/work/reports/feature_importance/data/correlation_plots/problem_space/min_by_mean_per_problem.png')

plt.show()