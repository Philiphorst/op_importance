import numpy as np
import matplotlib.pyplot as plt
import modules.misc.PK_helper as hlp
import modules.feature_importance.PK_ident_top_op as idtop
import modules.feature_importance.PK_plot_functions as fiplt

 
intermediate_data_root = '../data/'
all_classes_avg_out_path = intermediate_data_root+'/all_classes_avg.npy'
op_id_good_path = intermediate_data_root +'/op_id_good.npy'
op_id_order_path = intermediate_data_root +'/op_id_order.npy'
sort_good_ind_path = intermediate_data_root +'sort_good_ind.npy'

# -- Load the data
all_classes_avg = np.load(all_classes_avg_out_path)
op_id_good = np.load(op_id_good_path)

# -- Mask NaN entires
all_classes_avg_good = np.ma.masked_invalid(all_classes_avg[:,op_id_good])

# -- load a reference HCTSA_loc.mat containing all op_ids
import modules.misc.PK_matlab_IO as mIO    
op_ref_HCTSA_path = '/home/philip/work/OperationImportanceProject/results/done/HCTSA_Beef.mat'
op, = mIO.read_from_mat_file(op_ref_HCTSA_path,['Operations'],is_from_old_matlab = True)   

max_feat = 50
# -- calculate the correlation
abs_corr_array,sort_good_ind,all_classes_avg_good_norm = idtop.calc_perform_corr_mat(all_classes_avg_good,norm='z-score', max_feat = max_feat)


# -- save the op id's in order of performance (first entry = best performance)
np.save(op_id_order_path,op_id_good[sort_good_ind])
# -- sort the permutation vector that would sort the data array containing the good operations only 
np.save(sort_good_ind_path,sort_good_ind)

# -- extract the top feature names
names = hlp.ind_map_subset(op['id'], op['name'], op_id_good[sort_good_ind][:max_feat])



# -- Calculate the measures to be plotted
problems_succ = (~all_classes_avg_good[:,sort_good_ind[:max_feat]].mask).sum(axis=0)
u_stat_mean = all_classes_avg_good_norm[:,sort_good_ind[:max_feat]].mean(axis=0)

measures = np.vstack((problems_succ,u_stat_mean))

fiplt.plot_arr_dendrogram(abs_corr_array,names,measures = measures)
plt.savefig('/home/philip/Desktop/tmp/figure_tmp/corr_array.png')

plt.show()
