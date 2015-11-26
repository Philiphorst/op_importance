import numpy as np
import Task
import Data_Input
import Feature_Stats
import collections
import modules.misc.PK_helper as hlp

class Workflow:
    
    def __init__(self,task_names,input_method,stats_method):
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
        """
        self.task_names = task_names
        self.input_method = input_method
        self.stats_method = stats_method        

        # -- list of Tasks for this workflow
        self.tasks = [Task.Task(task_name,self.input_method,self.stats_method) for task_name in task_names]

        # -- Counter array for number of problems calculated successfully for each operation
        
        # -- place holders 
        self.good_op_ids = []
        self.stats_good_op = None
  
        
        
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
            if c[key] > threshold:
                self.good_op_ids.append(key)
                
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
                stats_good_op_ma_tmp = np.empty((len(self.good_op_ids),task.tot_stats.shape[0]))
            else:
                stats_good_op_ma_tmp = np.empty((len(self.good_op_ids)))
            stats_good_op_ma_tmp[:] = np.NaN
            
            ind = hlp.ismember(task.op_ids,self.good_op_ids,is_return_masked_array = True,return_dtype = int)
            # -- it is position in task.op_ids and i is position in self.good_op_ids
            for it,i in enumerate(ind):
                if i is not np.ma.masked: # -- that means the entry in task.op_ids is also in self.good_op_ids
                    stats_good_op_ma_tmp[i] = task.tot_stats[it].T
            # -- We return to the usual ordering: column equals feature
            stats_good_op_tmp.append(stats_good_op_ma_tmp.T)
        self.stats_good_op = np.vstack(stats_good_op_tmp)
        
    
    
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
            
    def calculate_stats(self):
        """
        Calculate the statistics of the features for each task using the method given by stats_method
        """
        for task in self.tasks:
            task.calc_stats(is_keep_data = False)
            
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
            
            
            
if __name__ == '__main__':

    path_pattern = '/home/philip/work/OperationImportanceProject/results/reduced/HCTSA_{:s}_N_70_100_reduced.mat'
    path_pattern_task_attrib = "../data/intermediate_results/task_{:s}_{:s}"

    masking_method = 'NaN'
    label_regex_pattern = '.*,(.*)$'
    task_names = ['Lighting2','OliveOil']
    combine_pair_method = 'mean'
        
    input_method = Data_Input.Datafile_Input(path_pattern,masking_method,label_regex_pattern)
    ranking_method = Feature_Stats.U_Stats(combine_pair_method)
    
    workflow = Workflow(task_names,input_method,ranking_method)

    if False:
        workflow.read_data()
        workflow.calculate_stats()
        workflow.save_task_attribute('tot_stats', path_pattern_task_attrib)
    else:
        workflow.read_data(is_read_feature_data = False)
        workflow.load_task_attribute('tot_stats', path_pattern_task_attrib)
        
    workflow.find_good_op_ids(1)
    
    workflow.collect_stats_good_op_ids()
    print workflow.stats_good_op
    print workflow.stats_good_op.shape