import numpy as np

class Task:
    
    def __init__(self,name,input_method,stat_method=None):
        """
        Constructor.
        Parameters:
        -----------
        name : str
            The name of the current task
        input_method : Data_Input
            The data input method used to read the data from disk. 
        stat_method : Feature_Stats
            The mehtod used to calculate the statistics
        """
        # -- calculation methods
        self.input_method = input_method
        self.stat_method = stat_method
        # -- task identifier
        self.name = name
        
        # -- data placeholders
        self.labels = np.array([])
        self.data = np.ma.masked_array([])
        self.op_ids = np.array([])
        self.pair_stats = np.ma.masked_array([])
        self.tot_stats = np.ma.masked_array([])
               
    def calc_stats(self,is_keep_data = False):
        """
        Calculate the statistics using the method given by self.stat_method. Pairwise and task total. 
        Parameters:
        ----------
        is_keep_data : bool
            Is the feature data to be kept after calculating the statistics or discarded (to save RAM space)?
            
        """
        self.pair_stats = self.stat_method.calc_pairs(self.labels,self.data)
        
        # -- free data if not required anymore to safe RAM space
        if not is_keep_data:
            self.data = None
        # -- combine the stats of the label pairs to one pooled stat for each feature
        self.tot_stats = self.stat_method.combine_pair(self.pair_stats)
        
    def read_data(self,is_read_feature_data = True):
        """
        Read the data using the input method given by self.input_method.
        Paramters:
        ----------
        is_read_feature_data : bool
            Is the feature data to be read or not
        """
        self.data, keywords_ts, self.op_ids = self.input_method.input_task(self.name,is_read_feature_data = is_read_feature_data)
        self.labels = self.input_method.extract_labels(keywords_ts)
    
    def load_attribute(self,attribute_name,in_path_pattern):
        """
        Load an attribute of the instance from a file
        Parameters:
        -----------
        attribute_name : string
            The name of the attribute of Task to be loaded
        out_path-pattern : string
            A string containing the pattern for the path pointing to the input file. 
            Formatted as in_path_pattern.format(self.name,attribute_name) + file extension
        """
       
        if attribute_name == 'tot_stats':  
            self.tot_stats = np.ma.load(in_path_pattern.format(self.name,attribute_name)+'.pckl')      
    
    
    def save_attribute(self,attribute_name,out_path_pattern):    
        """
        Save an attribute of the instance to a file
        Parameters:
        -----------
        attribute_name : string
            The name of the attribute of Task to be saved
        out_path-pattern : string
            A string containing the pattern for the path pointing to the output file. 
            Formatted as out_path_pattern.format(self.name,attribute_name) + file extension
        """ 
        if attribute_name == 'tot_stats':  
            np.ma.dump(self.tot_stats, out_path_pattern.format(self.name,attribute_name)+'.pckl') 
                 
           
