import numpy as np
import Task
import Data_Input
import Feature_Stats
import Reducing_Redundancy
import Plotting

import collections
import modules.misc.PK_helper as hlp
import modules.feature_importance.PK_feat_array_proc as fap

import matplotlib.pyplot as plt

class Workflow:
    
    def __init__(self,task_names,input_method,stats_method,redundancy_method,combine_tasks_method = 'mean',
                 combine_tasks_norm = None,
                 select_good_perf_ops_method = 'sort_asc',
                 select_good_perf_ops_norm = 'zscore',
                 n_good_perf_ops = None):
        """
        Constructor
        Parameters:
        -----------
        task_names : list of str
            A list of task names to be included in this workflow
        input_method : Data_Input
            The data input method used to read the data from disk. 
        stat_method : Feature_Stats
            The mehtod used to calculate the statistics
        redundancy_method : Reducing_Redundancy   
            The method used to reduce the redundancy in the well performing features
        combine_tasks_method : str
            The name describing the method used to combine the statistics for each task to create a single 1d arrray with 
            a single entry for every operation
        combine_tasks_norm : str
            The name of the normalisation method applied to the stats of each task before the statistics for each task are combined
        select_good_perf_ops_method : str
            The name describing the method used to sort the operations so the best operations come first in the self.stats_good_perf_op_comb
             and self.good_perf_op_ids
        select_good_perf_ops_norm : str
            The name describing the norm used when combining all statistics for all tasks for each operations
        self.n_good_perf_op_ids : int, optional
            Maximum entries in self.stats_good_perf_op_comb and self.good_perf_op_ids. If None, all good operations are used.
        """
        self.task_names = task_names
        self.input_method = input_method
        self.stats_method = stats_method        
        self.redundancy_method = redundancy_method
        self.combine_tasks_norm = combine_tasks_norm
        #normalise_array(data,axis,norm_type = 'zscore')
        if combine_tasks_method == 'mean':
            self.combine_tasks = self.combine_task_stats_mean
            
        if combine_tasks_norm == 'zscore':
            self.combine_task_norm_method = lambda y : fap.normalise_masked_array(y,axis = 1,norm_type = 'zscore')[0]
        else:
            # -- no normalisation - id-function
            self.combine_task_norm_method = lambda y : y

        if select_good_perf_ops_method == 'sort_asc':
            self.select_good_perf_ops = self.select_good_perf_ops_sort_asc
        self.select_good_perf_ops_norm = select_good_perf_ops_norm
        
        self.n_good_perf_ops = n_good_perf_ops 
        # -- list of Tasks for this workflow
        self.tasks = [Task.Task(task_name,self.input_method,self.stats_method) for task_name in task_names]

        # -- Counter array for number of problems calculated successfully for each operation
        
        # -- place holders 
        self.good_op_ids = []
        self.stats_good_op = None
        self.stats_good_op_comb = None
        self.stats_good_perf_op_comb = None
        self.good_perf_op_ids = None
                   
    def calculate_stats(self,is_keep_data = False):
        """
        Calculate the statistics of the features for each task using the method given by stats_method
        """
        for task in self.tasks:
            task.calc_stats(is_keep_data = is_keep_data)     
                      
    def collect_stats_good_op_ids(self):
        """
        Collect all combined stats for each task and take stats for good operations only
        """
        #stats_good_op_ma = np.empty((data.shape[0],np.array(self.good_op_ids).shape[0]))
        stats_good_op_tmp = []
        #stats_good_op_ma[:] = np.NaN
        for task in self.tasks:
            # -- create tmp array for good stats for current task. For sake of simplicity when dealing with different
            # dimensions of task.tot_stats we transpose stats_good_op_ma_tmp so row corresponds to feature temporarily
            if task.tot_stats.ndim > 1:
                stats_good_op_ma_tmp = np.empty((self.good_op_ids.shape[0],task.tot_stats.shape[0]))
            else:
                stats_good_op_ma_tmp = np.empty((self.good_op_ids.shape[0]))
            stats_good_op_ma_tmp[:] = np.NaN
            
            ind = hlp.ismember(task.op_ids,self.good_op_ids,is_return_masked_array = True,return_dtype = int)
            # -- it is position in task.op_ids and i is position in self.good_op_ids
            for it,i in enumerate(ind):
                if i is not np.ma.masked: # -- that means the entry in task.op_ids is also in self.good_op_ids
                    stats_good_op_ma_tmp[i] = task.tot_stats[it].T
            # -- We return to the usual ordering: column equals feature
            stats_good_op_tmp.append(stats_good_op_ma_tmp.T)
        self.stats_good_op = np.ma.masked_invalid(np.vstack(stats_good_op_tmp))
        
        
    def combine_task_stats_mean(self):
        """
        Combine the stats of all the tasks using the average over all tasks
        
        """
        self.stats_good_op_comb = self.combine_task_norm_method(self.stats_good_op).mean(axis=0)
        
    def init_redundancy_method_problem_space(self):
        """
        Initialises the redundancy_method with the required parameters depending on the redundancy_method.compare_space
        """
        if self.redundancy_method.compare_space == 'problem_stats':
            self.redundancy_method.set_parameters(self.stats_good_op,self.good_op_ids,self.good_perf_op_ids)
        
           
    def find_good_op_ids(self, threshold):
        """
        Find the features that have been successfully calculated for more then threshold problems.
        Parameters:
        -----------
        threshold : int
            Only keep features that have been calculated for at least threshold tasks.
        
        """
        # -- List of all op_ids for each task (with duplicates)
        op_ids_tasks = [item for task in self.tasks for item in task.op_ids.tolist()]

        c = collections.Counter(op_ids_tasks)
        for key in c.keys():
            if c[key] >= threshold:
                self.good_op_ids.append(key) 
        self.good_op_ids = np.array(self.good_op_ids) 
                                  
    def load_task_attribute(self,attribute_name,in_path_pattern): 
        """
        Load an attribute for all tasks from separate files
        Parameters:
        -----------
        attribute_name : string
            The name of the attribute of Task to be saved
        out_path-pattern : string
            A string containing the pattern for the path pointing to the input file. Formatted as in_path_pattern.format(self.name,attribute_name)
        """      
        for task in self.tasks:
            task.load_attribute(attribute_name,in_path_pattern)
    
    def read_data(self,is_read_feature_data = True): 
        """
        Read the data for all tasks from disk using the method given by self.input_method 
        Parameters:
        -----------
        is_read_feature_data : bool
            Is the feature data to be read
        """   
        for task in self.tasks:
            task.read_data(is_read_feature_data = is_read_feature_data)           
    
    def save_task_attribute(self,attribute_name,out_path_pattern):
        """
        Save an attribute of of all tasks to separate files
        Parameters:
        -----------
        attribute_name : string
            The name of the attribute of Task to be saved
        out_path-pattern : string
            A string containing the pattern for the path pointing to the output file. Formatted as out_path_pattern.format(self.name,attribute_name)
        """ 
        for task in self.tasks:
            task.save_attribute(attribute_name,out_path_pattern)
            
    def select_good_perf_ops_sort_asc(self):
        """
        Select a subset of well performing operations
        """
                
        if self.select_good_perf_ops_norm in ['z-score','zscore'] :
            all_classes_good_norm = fap.normalise_masked_array(self.stats_good_op,axis = 1,norm_type = 'zscore')[0]
        
        elif self.select_good_perf_ops_norm == 'mean-norm':
            all_classes_good_mean = np.ma.masked_invalid(np.ma.mean(self.stats_good_op,axis = 1))
            all_classes_good_norm = (self.stats_good_op.T / all_classes_good_mean).T  
        
        else:
            all_classes_good_norm =  self.stats_good_op

        sort_ind_tmp = np.argsort(all_classes_good_norm.mean(axis=0))
        
        if self.n_good_perf_ops == None:
            self.stats_good_perf_op_comb  = self.stats_good_op_comb[sort_ind_tmp]
            self.good_perf_op_ids =  self.good_op_ids[sort_ind_tmp]
        else:
            self.stats_good_perf_op_comb  = self.stats_good_op_comb[sort_ind_tmp][:self.n_good_perf_ops]            
            self.good_perf_op_ids =  self.good_op_ids[sort_ind_tmp][:self.n_good_perf_ops]            

        
if __name__ == '__main__':
    
    # -----------------------------------------------------------------
    # -- Set Parameters -----------------------------------------------
    # -----------------------------------------------------------------
    path_pattern = '/home/philip/work/OperationImportanceProject/results/reduced/HCTSA_{:s}_N_70_100_reduced.mat'
    path_pattern_task_attrib = "../data/intermediate_results/task_{:s}_{:s}"
    plot_out_path = '/home/philip/Desktop/tmp/figure_tmp/test.png'
    result_txt_outpath = '/home/philip/Desktop/tmp/figure_tmp/result_txt.txt'
    
    
    masking_method = 'NaN'
    label_regex_pattern = '.*,(.*)$'
    #task_names = ['Lighting2','OliveOil','FaceFour','N', 'FISH']
    
    task_names = ['MedicalImages', 'Cricket_X', 'InlineSkate', 'ECG200', 'WordsSynonyms', 'uWaveGestureLibrary_X', 'Two_Patterns', 'yoga', 'Symbols', 'uWaveGestureLibrary_Z', 'SonyAIBORobotSurfaceII', 'Cricket_Y', 'Gun_Point', 'OliveOil', 'Lighting7', 'NonInvasiveFatalECG _Thorax1', 'Haptics', 'Adiac', 'ChlorineConcentration', 'synthetic_control', 'OSULeaf', 'DiatomSizeReduction', 'SonyAIBORobotSurface', 'MALLAT', 'uWaveGestureLibrary_Y', 'N', 'CBF', 'ECGFiveDays', 'Lighting2', 'FISH', 'FacesUCR', 'FaceFour', 'Trace', 'Coffee', '50words', 'MoteStrain', 'wafer', 'Cricket_Z', 'SwedishLeaf']

    combine_pair_method = 'mean'
    combine_tasks_method = 'mean'   
    combine_tasks_norm = None
    select_good_perf_ops_norm = 'zscore'
    select_good_perf_ops_method = 'sort_asc'
    similarity_method = 'correlation'
    compare_space = 'problem_stats'
    n_good_perf_ops = 50
    min_calc_tasks = 32
    
    # -----------------------------------------------------------------
    # -- Initialise Class instances -----------------------------------
    # -----------------------------------------------------------------
    
    input_method = Data_Input.Datafile_Input(path_pattern,masking_method,label_regex_pattern)
    ranking_method = Feature_Stats.U_Stats(combine_pair_method)
    redundancy_method = Reducing_Redundancy.Reducing_Redundancy(similarity_method = similarity_method,compare_space = compare_space)
    
    workflow = Workflow(task_names,input_method,ranking_method,
                        combine_tasks_method = combine_tasks_method,combine_tasks_norm = combine_tasks_norm,
                        select_good_perf_ops_method = select_good_perf_ops_method, select_good_perf_ops_norm = select_good_perf_ops_norm,
                        redundancy_method = redundancy_method,
                        n_good_perf_ops = n_good_perf_ops)
    
    # -----------------------------------------------------------------
    # -- Do the statistic calculations --------------------------------
    # -----------------------------------------------------------------    
    
    # -- calculate the statistics
    if False:
        workflow.read_data()
        workflow.calculate_stats()
        workflow.save_task_attribute('tot_stats', path_pattern_task_attrib)
    else:
        workflow.read_data(is_read_feature_data = False)
        workflow.load_task_attribute('tot_stats', path_pattern_task_attrib)
    
    # -- find the features which are calculated for at least min_calc_tasks tasks
    workflow.find_good_op_ids(min_calc_tasks)
    # -- Collect all combined stats for each task and take stats for good operations only
    workflow.collect_stats_good_op_ids()
    # -- Combine the stats of all the tasks 
    workflow.combine_tasks()
    # -- Select a subset of well performing operations
    workflow.select_good_perf_ops()
    
    # -----------------------------------------------------------------
    # -- Do the redundancy calculations -------------------------------
    # -----------------------------------------------------------------     
    
    # -- intitialise the redundancy method with the calculated results
    workflow.init_redundancy_method_problem_space()
    # -- calculate the correlation matrix saved in workflow.redundancy_method.similarity_array
    workflow.redundancy_method.calc_similarity()
    # -- calculate the linkage, the cluster indices and the clustering in self.corr_linkage,self.cluster_inds,self.cluster_op_id_list,respectively
    workflow.redundancy_method.calc_hierch_cluster()
    
    # -----------------------------------------------------------------
    # -- Do the plotting ----------------------------------------------
    # -----------------------------------------------------------------     
    # -- initialise the plotting class
    plotting = Plotting.Plotting(workflow)
    if True:
        # -- Plot the statistics array
        plotting.plot_stat_array()
    else:
        # -- Plot the similarity array   
        plotting.plot_similarity_array()
    
    plt.savefig(plot_out_path)
    plt.show()
    
    
    # -----------------------------------------------------------------
    # -- Output the results to text file-------------------------------
    # -----------------------------------------------------------------       
    op_id_name_map = plotting.map_op_id_name_mult_task(workflow.tasks)
    
    # -- write not reduced top performing features to text file
    with open(result_txt_outpath,'wb') as out_result_txt_file:
        for op_id,op_name,op_U in zip(workflow.good_perf_op_ids,
                                      hlp.ind_map_subset(op_id_name_map[0],op_id_name_map[1], workflow.good_perf_op_ids),
                                      workflow.stats_good_perf_op_comb):
            out_result_txt_file.write("{:d} {:s} {:f}\n".format(op_id,op_name,op_U))

    
    
    
    
    
    
    