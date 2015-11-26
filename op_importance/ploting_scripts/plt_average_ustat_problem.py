import numpy as np
import matplotlib.pyplot as plt


intermediate_data_root = '../data/'
all_classes_avg_out_path = intermediate_data_root+'/all_classes_avg.npy'
op_id_good_path = intermediate_data_root +'/op_id_good.npy'


# -- Load the data
all_classes_avg = np.load(all_classes_avg_out_path)
op_id_good = np.load(op_id_good_path)
# -- Mask NaN entires
all_classes_avg_good = np.ma.masked_invalid(all_classes_avg[:,op_id_good])

plt.bar(np.arange(39)+0.5,all_classes_avg_good.mean(axis=1))
plt.xlabel('ID of problem')
plt.ylabel('Average U-Statistics (worst case 0.5 - to be checked)')
plt.savefig('/home/philip/work/reports/feature_importance/data/correlation_plots/problem_space/avg_ustat_problem.png')
plt.show()