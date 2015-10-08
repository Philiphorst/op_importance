import glob 
import numpy as np
import cPickle as pickle




# -- local modules modules
import modules.feature_importance.PK_feature_quality as fq
import modules.feature_importance.PK_feat_array_proc as fap
import modules.feature_importance.PK_test_stats as tstat
import modules.misc.PK_matlab_IO as mIO

# ---------------------------------------------------------------------------------
# -- Global parameter definition
# ---------------------------------------------------------------------------------

mat_file_root = '/home/philip/work/OperationImportanceProject/results/reduced/'
mat_file_root = '/home/philip/work/OperationImportanceProject/results/test/'

mat_file_paths = glob.glob(mat_file_root+'*.mat')


intermediate_data_root = '../data/'

# data_all_good_op_path = intermediate_data_root + '/data_all.npy'
# ind_for_good_op_ids_path = intermediate_data_root +'/ind_dict.pckl'

data_all_good_op_path = intermediate_data_root + '/data_ma_most.npy'
ind_for_good_op_ids_path = intermediate_data_root +'/ind_dict_ma_most.pckl'

good_op_ids_path = intermediate_data_root +'/good_op_id.pckl'
corr_feat_mask_path = intermediate_data_root +'/mask.npy'
abs_corr_array_path = intermediate_data_root + '/abs_corr_mat.npy'
all_classes_avg_path = intermediate_data_root + 'all_classes_avg.npy'
mat_file_names_path = intermediate_data_root + 'mat_file_names.npy'

all_classes_avg_non_corr_path = intermediate_data_root + 'all_classes_avg_non_corr.npy'
ind_dict_non_corr_path = intermediate_data_root + 'ind_dict_non_corr.pckl'

count_op_id_min = 1 # -- minimum number of successful calculations for operation in problems

# -- Are the HCTSA files calculated with old version of matlab code
IS_FROM_OLD_MATLAB = True

# -- What has to be done
COMPUTE_COMPLETE_DATA = True    
COMPUTE_FEATURE_CORRELATION = True
CALCULATE_U_STATS = True
CALCULATE_ONLY_NEW_U_STATS = False
CALCULATE_U_STATS_ALL_CLASSES_AVG = True
CALCULATE_BEST_FEATURES = True

# ---------------------------------------------------------------------------------
# -- Compute complete data array for good op_ids
# ---------------------------------------------------------------------------------
if COMPUTE_COMPLETE_DATA:
    # -- calculate in how many problems each operation has been successfully calculated
    count_op_id = fq.count_op_calc(mat_file_root,is_from_old_matlab = IS_FROM_OLD_MATLAB)

    # -- pick only features calculated in all problems
    op_id_good = np.nonzero([count_op_id >= count_op_id_min])[1].tolist()

    # -- concatenate good features for all problems to one large feature matrix (can take a while and get big)
    data_all,ind_dict = fap.cat_data_op_subset(mat_file_paths,op_id_good,is_from_old_matlab = IS_FROM_OLD_MATLAB,is_do_dict_only = False)

    # -- safe the calculated values
    #np.save(data_all_good_op_path,data_all)
    np.ma.dump(data_all, data_all_good_op_path)
    pickle.dump(ind_dict,open(ind_for_good_op_ids_path,'w'))  
 
# ---------------------------------------------------------------------------------
# -- Compute the mask to remove correlated features
# ---------------------------------------------------------------------------------
if COMPUTE_FEATURE_CORRELATION:
    if not COMPUTE_COMPLETE_DATA:
        data_all = np.ma.load(data_all_good_op_path)
        print 'all data successfully loaded'
    # -- compute the correlated features and a mask that removes them
    mask,abs_corr_array = fap.corelated_features_mask(data=data_all,abs_corr_array=None) 
    np.save(corr_feat_mask_path,mask)
    np.save(abs_corr_array_path,abs_corr_array)

# ---------------------------------------------------------------------------------
# -- Calculate U_statistics for the problems
# ---------------------------------------------------------------------------------
if CALCULATE_U_STATS:
    
    # -- skip problems with already calculated U-stats
    if CALCULATE_ONLY_NEW_U_STATS:
        task_names = tstat.filter_calculated(mat_file_root)
        file_paths = [mat_file_root+"HCTSA_{0:s}_N_70_100_reduced.mat".format(s) for s in task_names]
    
    # -- calculate U-stats for all problems        
    else:
        file_paths = mat_file_paths
        _,task_names = tstat.get_calculated_names(mat_file_root)
        
    # -- do the actual calculation of U-stats and save to mat_file_root  
    for file_path,task_name in zip(file_paths,task_names):
        print 'Calculating U-statistics for {:s}.'.format(task_name)
        ranks = tstat.u_stat_all_label_file_name(file_path, is_from_old_matlab = IS_FROM_OLD_MATLAB )[0]
        np.save(mat_file_root+task_name+'_ustat.npy',ranks)

# ---------------------------------------------------------------------------------
# -- Calculate the average u score over all problems for each used feature
# ---------------------------------------------------------------------------------
if CALCULATE_U_STATS_ALL_CLASSES_AVG:
    
    # -- get the filenames for all problems under investigation 
    calced_names = tstat.get_calculated_names(mat_file_root)[0]
    u_stat_file_names = ["{0:s}_ustat.npy".format(s) for s in calced_names]
    mat_file_names = ["HCTSA_{0:s}_N_70_100_reduced.mat".format(s) for s in calced_names]
    
    # -- initialise the array containing the average u-statistic values for all problems and features
    all_classes_avg = np.ones((len(u_stat_file_names),10000))*np.NAN
    
    for i,(u_stat_file_name, mat_file_name) in enumerate(zip(u_stat_file_names,mat_file_names)):
        
        # -- load the u statistic for every operation and label pairing
        u_stat = np.load(mat_file_root+u_stat_file_name)

        # -- calculate the scaling factor for every label pairing of the current classification problem
        u_scale = tstat.u_stat_norm_factor(mat_file_root+mat_file_name,is_from_old_matlab = IS_FROM_OLD_MATLAB)

        # -- calculate the average scaled u statistic over all label pairs in current problem 
        u_stat_avg = (u_stat.T/u_scale).transpose().mean(axis=0)
        
        # -- save the average scaled u-statistic for all features to the all_classes_avg array. 
        #    The column number corresponds with the operation id
        op, = mIO.read_from_mat_file(mat_file_root+mat_file_name,['Operations'],is_from_old_matlab = IS_FROM_OLD_MATLAB )
        all_classes_avg[i,op['id']] = u_stat_avg
    
    np.save(all_classes_avg_path,all_classes_avg)
    np.save(mat_file_names_path,mat_file_names)
    
  
# ---------------------------------------------------------------------------------
# -- Calculate the best features 
# ---------------------------------------------------------------------------------
# all_classes_avg_non_corr_path = intermediate_data_root + 'all_classes_avg_non_corr_no_time_picking.npy'
# ind_dict_non_corr_path = intermediate_data_root + 'ind_dict_non_corr_no_time_picking.pckl'
# corr_feat_mask_path = intermediate_data_root +'/mask_no_time_picking.npy'
if CALCULATE_BEST_FEATURES:
    if not CALCULATE_U_STATS_ALL_CLASSES_AVG:
        all_classes_avg = np.load(all_classes_avg_path)
        # -- get the filenames for all problems under investigation 
        calced_names = tstat.get_calculated_names(mat_file_root)[0]
        u_stat_file_names = ["{0:s}_ustat.npy".format(s) for s in calced_names]
        mat_file_names = ["HCTSA_{0:s}_N_70_100_reduced.mat".format(s) for s in calced_names]
    if not COMPUTE_COMPLETE_DATA:
        #the indices for the good operations in every problem
        ind_dict = pickle.load(open(ind_for_good_op_ids_path,'r'))
    
    # -- load the mask of the correlated features in the concatenated data matrix
    corr_feat_mask = np.load(corr_feat_mask_path)
    # -- initilise the dictionary that holds indices and operation ids for the non correlated
    #    operations
    ind_dict_non_corr = dict()

    for key in ind_dict:
        ind_dict_non_corr[key] = ind_dict[key][:,corr_feat_mask]
        nr_non_corr_ops = ind_dict_non_corr[key].shape[1]
        
    all_classes_avg_non_corr = np.zeros((all_classes_avg.shape[0],nr_non_corr_ops))  
    
    for i,mat_file_name in enumerate(mat_file_names):
        mat_file_path = mat_file_root+mat_file_name
        # -- get the indices of the good operations in this problem
        op_ids_non_corr = ind_dict_non_corr[mat_file_path][1]
        # -- The average u scores are are saved with the op_id as column index of the entry
        all_classes_avg_non_corr[i] = all_classes_avg[i][op_ids_non_corr]
        
    # XXX What up with problem 30 ???
    print "finished: ", ind_dict_non_corr
    np.save(all_classes_avg_non_corr_path,all_classes_avg_non_corr)
    pickle.dump(ind_dict_non_corr,open(ind_dict_non_corr_path,'w'))

