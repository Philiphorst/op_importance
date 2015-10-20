"""
Use hierarchical clustering to build clusters of operations by
with the correlation as their pairwise distance measure

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
"""

# import numpy as np
# import cPickle as pickle
# import matplotlib.pyplot as plt
# import scipy.spatial.distance as spdst
# #import scipy.cluster.hierarchy as hierarchy
# import modules.misc.PK_matlab_IO as mIO
# import modules.feature_importance.PK_feat_array_proc as fap
# 
# # -- define the paths to data files
# intermediate_data_root = '/home/philip/work/Git_repositories/op_importance/op_importance/data/'
# abs_corr_array_path = intermediate_data_root + '/abs_corr_mat.npy'
# link_arr_path = intermediate_data_root + '/link_arr.npy'
# data_good_op_path = intermediate_data_root + '/data_ma_most.npy'
# 
# ind_for_good_op_ids_path = intermediate_data_root +'/ind_dict_ma_most.pckl'
# large_cluster_data_path = intermediate_data_root +'/large_cluster_data.npy'
# 
# CALCULATE_LINKAGE = False
# 
# if CALCULATE_LINKAGE:
#     # -- load the correlation matrix
#     abs_corr_array = np.load(abs_corr_array_path)
#      
#     # -- transform the correlation matrix into distance measure
#     abs_corr_dist_arr = np.around(1 - abs_corr_array,7)
#  
#     # -- transform the correlation matrix into condensed distance matrix
#     dist_corr = spdst.squareform(abs_corr_dist_arr)   
#     
#     # -- force calculation of linkage
#     is_force_calc_link_arr = True     
#    
# else:
#     # -- skip calculation and load linkage from link_arr_path
#     is_force_calc_link_arr = False     
#     abs_corr_dist_arr = None
#     
# cluster_lst, cluster_size_lst = fap.compute_clusters_from_dist(abs_corr_dist_arr=abs_corr_dist_arr,link_arr_path = link_arr_path, is_force_calc_link_arr = is_force_calc_link_arr)
# 
# 
# # ---------------------------------------------------------------------
# # -- Save clustr op names to textfile ---------------------------------
# # ---------------------------------------------------------------------
# 
# # -- map indices in abs_corr_array to op_ids
# ind_dict = pickle.load(open(ind_for_good_op_ids_path,'r'))
# hctsa_temp_path = ind_dict.keys()[0]
# hctsa_temp_inds = ind_dict[hctsa_temp_path]
# #-- get operation names
# op, = mIO.read_from_mat_file(hctsa_temp_path,['Operations'],is_from_old_matlab = True)
# op_names = np.array(op['name'])[hctsa_temp_inds[0]]
# with open('/home/philip/work/reports/feature_importance/data/operations_cluster.txt','w') as out_file:
#     
#     out_file.write('------------------------------------------------------------------\n')
#     out_file.write('--- clusters of operations----------------------------------------\n')
#     out_file.write('------------------------------------------------------------------\n')
#     for cluster in cluster_lst:
#         cluster.sort()
#         for op in cluster:
#             out_file.write(op_names[op]+'\n')
#         out_file.write('------------------------------------------------------------------\n')
# 
# # ---------------------------------------------------------------------
# # -- pick longest cluster for demonstration purposes
# if False:
#     max_cluster_lst_ind = cluster_size_lst.index(max(cluster_size_lst))
#     max_cluster_corr_inds =  cluster_lst[max_cluster_lst_ind]
# 
#     data = np.load(data_good_op_path)
#     data = data[:,max_cluster_corr_inds]
#     np.ma.dump(data, large_cluster_data_path)
# else:
#     data = np.ma.load(large_cluster_data_path)
# data = fap.normalise_array(data, axis = 0,norm_type ='sigmoid')[0]
# plt.matshow(data,aspect='auto')
# #plt.colorbar()
# plt.show()