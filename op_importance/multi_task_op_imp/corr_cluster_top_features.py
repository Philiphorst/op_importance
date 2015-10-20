import numpy as np
import matplotlib.pyplot as plt
import modules.misc.PK_helper as hlp
import modules.feature_importance.PK_feat_array_proc as fap

import scipy.spatial.distance as spdst

# -- load the ustat for all tasks
intermediate_data_root = '../data/'
op_id_data_column_path = intermediate_data_root + '/op_id_good.npy'
all_classes_avg_out_path = intermediate_data_root+'/all_classes_avg.npy'
data_all_good_op_path = intermediate_data_root + '/data_all.npy'
data_top_op_path = intermediate_data_root + '/data_reduced.npy'
abs_corr_array_path = intermediate_data_root + '/abs_corr_array.npy'
link_arr_path = intermediate_data_root + '/link_arr.npy'


IS_LOAD_ALL_DATA = False
IS_CALCULATE_CORRELATION_MATRIX = False
IS_CALCULATE_LINKAGE = True

all_classes_avg = np.load(all_classes_avg_out_path)

# -- calculate the mean u stat over all tasks
# -- The op_id for each entry of mean_ustat[i] is i
mean_ustat = np.nanmean(all_classes_avg,axis=0)

# -- load the mapping from good op_ids to indices of the data matrix
# -- For every entry op_id_good[i] is the op_id for the entry i in the data matrix
op_id_data_column = np.load(op_id_data_column_path)

# -- get the u stats for the operations that are good enough quality
# -- The op_id for each entry mean_ustat_data_column[i] is op_id_good[i] which is also the
# -- op_id for the entry i in the data matrix
mean_ustat_data_column = mean_ustat[op_id_data_column]

# -- get the op _ids with the lowest average u stat
# -- sort_ind[i] is column in data of i-th best operation
sort_ind = np.argsort(mean_ustat_data_column)
sorted_op_ids = op_id_data_column[sort_ind]

# -- Load all data
if IS_LOAD_ALL_DATA:
    # -- load data for all tasks
    data_all = np.load(data_all_good_op_path)
    # -- sort the data by ustat and consider top operations only
    data_top_op = data_all[:,sort_ind][:,:200]
    # -- save reduced data to disk
    np.save(data_top_op_path,data_top_op)
else:
    # -- load reduced data
    data_top_op = np.load(data_top_op_path)
    print data_top_op.shape
    
# -- calculate correlation matrix
if IS_CALCULATE_CORRELATION_MATRIX:
    # -- Create masked array from reduced data
    data_top_op = np.ma.masked_invalid(data_top_op)[:,:200]
    
    # -- calculate correlation matrix
    abs_corr_array = np.abs(np.ma.corrcoef(data_top_op, rowvar=0))  
    
    np.save(abs_corr_array_path,abs_corr_array.data)
else:
    abs_corr_array = np.load(abs_corr_array_path)
    print abs_corr_array.shape
# -- calculate linkage and clusters
   


if IS_CALCULATE_LINKAGE:
    # -- load the correlation matrix
    abs_corr_array = np.load(abs_corr_array_path)
     
    # -- transform the correlation matrix into distance measure
    abs_corr_dist_arr = np.around(1 - abs_corr_array,7)
 
    # -- transform the correlation matrix into condensed distance matrix
    dist_corr = spdst.squareform(abs_corr_dist_arr)   
    
    # -- force calculation of linkage
    is_force_calc_link_arr = True     
   
else:
    # -- skip calculation and load linkage from link_arr_path
    is_force_calc_link_arr = False     
    abs_corr_dist_arr = None

# -- cluster of indices in abs_corr_dist_arr array 
cluster_lst, cluster_size_lst = fap.compute_clusters_from_dist(abs_corr_dist_arr=abs_corr_dist_arr,link_arr_path = link_arr_path, is_force_calc_link_arr = is_force_calc_link_arr)   

# -- cluster of operation ids
op_id_cluster = [[sorted_op_ids[ind] for ind in cluster] for cluster in cluster_lst]




# -- load a reference HCTSA_loc.mat containing all op_ids
import modules.misc.PK_matlab_IO as mIO
op_ref_HCTSA_path = '/home/philip/work/OperationImportanceProject/results/done/HCTSA_Beef.mat'
op, = mIO.read_from_mat_file(op_ref_HCTSA_path,['Operations'],is_from_old_matlab = True)


# -- create the clusters of information
tuple_cluster_lst = []
for cluster in cluster_lst:
    tuple_cluster = []
    for ind in cluster:
        # -- get the index for the imported hctsa_data array
        perc_success = np.count_nonzero(~np.isnan(data_top_op[:,ind]))/float(data_top_op.shape[0])*100
        op_id_curr = sorted_op_ids[ind]
        ind_hctsa_loc_curr = op['id'].index(op_id_curr)
        # -- add all the information of current op_id/ind
        tuple_curr = (op['name'][ind_hctsa_loc_curr],op_id_curr,op['master_id'][ind_hctsa_loc_curr],mean_ustat[op_id_curr],perc_success)
        tuple_cluster.append(tuple_curr)                                                                                                                                                                                                                                                                                                                                                           
    tuple_cluster_lst.append(tuple_cluster)
    
with open('/home/philip/work/reports/feature_importance/data/operations_cluster.txt','w') as out_file:
    
    out_file.write('------------------------------------------------------------------\n')
    out_file.write('--- clusters of operations----------------------------------------\n')
    out_file.write('------------------------------------------------------------------\n\n\n\n')
    for tuple_cluster in tuple_cluster_lst:
        for tuple_curr in tuple_cluster:
            out_file.write('{:s}\t{:d}\t{:d}\t{:f}\t{:f}\n'.format(*tuple_curr))
        out_file.write('------------------------------------------------------------------\n\n')

