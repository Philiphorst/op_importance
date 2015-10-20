'''
Created on 1 Jul 2015

@author: philip knaute
------------------------------------------------------------------------------
Copyright (C) 2015, Philip Knaute <philiphorst.project@gmail.com>,

This work is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of
this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/ or send
a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
California, 94041, USA.
------------------------------------------------------------------------------
'''

import glob 
import numpy as np

# -- local modules modules
import modules.feature_importance.PK_feat_array_proc as fap
import modules.feature_importance.PK_test_stats as tstat

# ---------------------------------------------------------------------------------
# -- Global parameter definition
# ---------------------------------------------------------------------------------

mat_file_root = '/home/philip/work/OperationImportanceProject/results/reduced/'
#mat_file_root = '/home/philip/work/OperationImportanceProject/results/test/'

mat_file_paths = glob.glob(mat_file_root+'*.mat')
# -- check if there are suitable files in the given mat_file_folder
if mat_file_paths == []:
    print "No suitable data in {:s}".format(mat_file_root)
    exit()
    
intermediate_data_root = '../data/'


data_all_good_op_path = intermediate_data_root + '/data_all.npy'
op_id_good_path = intermediate_data_root +'/op_id_good.npy'

ustat_data_out_folder = intermediate_data_root
all_classes_avg_out_path = intermediate_data_root+'/all_classes_avg.npy'


count_op_id_min = 32 # -- minimum number of successful calculations for operation in problems

# -- Are the HCTSA files calculated with old version of matlab code
IS_FROM_OLD_MATLAB = True

# -- What has to be done
COMPUTE_COMPLETE_DATA = True 
CALCULATE_U_STATS = True
CALCULATE_ONLY_NEW_U_STATS = False
CALCULATE_U_STATS_ALL_CLASSES_AVG = True
#CALCULATE_BEST_FEATURES = False

# ---------------------------------------------------------------------------------
# -- Compute complete data array for good op_ids
# ---------------------------------------------------------------------------------
    
if COMPUTE_COMPLETE_DATA:
    data_all,op_id_good = fap.cat_data_from_matfile_root(mat_file_paths, count_op_id_min,is_from_old_matlab = IS_FROM_OLD_MATLAB,
                               data_all_good_op_path = data_all_good_op_path,op_id_good_path = op_id_good_path,is_return_masked = False)

# -- Create masked array from data_all    
# data_all = np.ma.masked_invalid(data_all)

# ---------------------------------------------------------------------------------
# -- Calculate U_statistics for the problems
# ---------------------------------------------------------------------------------   
if CALCULATE_U_STATS:
    
    # -- skip problems with already calculated U-stats
    if CALCULATE_ONLY_NEW_U_STATS:
        task_names = tstat.filter_calculated(mat_file_root,HCTSA_name_search_pattern = 'HCTSA_(.*)_N_70_100_reduced.mat')
        file_paths = [mat_file_root+"HCTSA_{0:s}_N_70_100_reduced.mat".format(s) for s in task_names]
    
    # -- calculate U-stats for all problems        
    else:
        file_paths = mat_file_paths
        _,task_names = tstat.get_calculated_names(mat_file_root,HCTSA_name_search_pattern = 'HCTSA_(.*)_N_70_100_reduced.mat')
    
    u_stat_file_paths = tstat.calculate_ustat_mult_tasks(mat_file_paths,task_names,ustat_data_out_folder,is_from_old_matlab = IS_FROM_OLD_MATLAB)  
    if CALCULATE_U_STATS_ALL_CLASSES_AVG:
        all_classes_avg = tstat.calculate_ustat_avg_mult_task(mat_file_paths,u_stat_file_paths,all_classes_avg_out_path ,is_from_old_matlab = IS_FROM_OLD_MATLAB)


  
  
# ---------------------------------------------------------------------------------
# -- Calculate the best features 
# ---------------------------------------------------------------------------------
# all_classes_avg_non_corr_path = intermediate_data_root + 'all_classes_avg_non_corr_no_time_picking.npy'
# ind_dict_non_corr_path = intermediate_data_root + 'ind_dict_non_corr_no_time_picking.pckl'
# corr_feat_mask_path = intermediate_data_root +'/mask_no_time_picking.npy'

# if CALCULATE_BEST_FEATURES:
#     if not CALCULATE_U_STATS_ALL_CLASSES_AVG:
#         all_classes_avg = np.load(all_classes_avg_path)
#         # -- get the filenames for all problems under investigation 
#         calced_names = tstat.get_calculated_names(mat_file_root)[0]
#         u_stat_file_names = ["{0:s}_ustat.npy".format(s) for s in calced_names]
#         mat_file_names = ["HCTSA_{0:s}_N_70_100_reduced.mat".format(s) for s in calced_names]
#     if not COMPUTE_COMPLETE_DATA:
#         # -- the indices for the good operations in every problem
#         ind_dict = pickle.load(open(ind_for_good_op_ids_path,'r'))
#     
#     # -- load the mask of the correlated features in the concatenated data matrix
#     corr_feat_mask = np.load(corr_feat_mask_path)
#     # -- initilise the dictionary that holds indices and operation ids for the non correlated
#     #    operations
#     ind_dict_non_corr = dict()
# 
#     for key in ind_dict:
#         ind_dict_non_corr[key] = ind_dict[key][:,corr_feat_mask]
#         nr_non_corr_ops = ind_dict_non_corr[key].shape[1]
#         
#     all_classes_avg_non_corr = np.zeros((all_classes_avg.shape[0],nr_non_corr_ops))  
#     
#     for i,mat_file_name in enumerate(mat_file_names):
#         mat_file_path = mat_file_root+mat_file_name
#         # -- get the indices of the good operations in this problem
#         op_ids_non_corr = ind_dict_non_corr[mat_file_path][1]
#         # -- The average u scores are are saved with the op_id as column index of the entry
#         all_classes_avg_non_corr[i] = all_classes_avg[i][op_ids_non_corr]
#         
#     # XXX What up with problem 30 ???
#     print "finished: ", ind_dict_non_corr
#     np.save(all_classes_avg_non_corr_path,all_classes_avg_non_corr)
#     pickle.dump(ind_dict_non_corr,open(ind_dict_non_corr_path,'w'))

