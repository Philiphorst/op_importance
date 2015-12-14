import modules.misc.PK_matlab_IO as mIO
import re
import numpy as np


class Data_Input:
    def __init__(self,masking_method,label_regex_pattern = '.*,(.*)$'):  
        """ 
        Constructor
        Parameters:
        -----------
        masking_method : string
            String describing method used for masking invalid entries
        label_regex_pattern : string, optional          
            The pattern used to extract the label from the keywords. The label is
            the first group of the regular expression given 
        Returns:
        --------
            None
        """
        
        self.regex_pattern = label_regex_pattern
        # -- set the correct method for masking not calculated operations
        if masking_method in ['nan','NaN','NAN']:
            self.masking_method = self.mask_data_invalid
    
    def extract_labels(self, keywords_ts, regex_pattern = None):
        """
        Extract the labels from a list/array of keywords
        Parameters:
        -----------
        keywords_ts : list/ndarray
            Array like containing the keywords for the current timeseries 
        regex_pattern : string, optional
            The pattern used to extract the label from the keywords_ts. The label is
            the first group of the regular expression given
        Returns:
        --------
            1-D array containing the labels for the current timeseries
        """
        
        if regex_pattern == None:
            regex_pattern = self.regex_pattern
        # -- extract the labels for the rows from the timeseries keywords
        reg_ex = re.compile(regex_pattern) # --  pick the substring after the last comma
        # FIXME some sort of error handling would be useful 
        labels = np.array([reg_ex.match(keywords).group(1) for keywords in keywords_ts])

        return labels
   
#     def remove_bad_ops(self,data,op_id,threshold = 1.):
#         """
#         Remove all operastions with a fail rate of threshold or more.
#         """
    def mask_data_invalid(self,data):
        """
        Mask not calculated features using by masking all NaN entries
        Parameters:
        -----------
        data : ndarray
            Array containing the features for all timeseries
        Returns:
        --------
        data : masked array
            Array containing the features for all timeseries with NaN entries masked            
        """
        return np.ma.masked_invalid(data)

    
    
    
class Datafile_Input(Data_Input):
    
    def __init__(self, path_pattern,masking_method = 'nan',label_regex_pattern = '.*,(.*)$'):
        """
        Constructor
        Parameters:
        -----------
        path_pattern : string
            Pattern used to construct the path to the current matlab file.
        masking_method : string
            String describing method used for masking invalid entries
        label_regex_pattern : string, optional
            The pattern used to extract the label from the keywords_ts. The label is
            the first group of the regular expression given
        Returns:
        -------
            None
                    
        """
        Data_Input.__init__(self,masking_method,label_regex_pattern = '.*,(.*)$')
        # -- initialise the pattern for the home folder of the data files
        self.path_pattern = path_pattern
        
    def input_task(self,task_name,is_read_feature_data = True):
        """
        Read the required data from a HCTSA_loc.mat file
        Parameters:
        -----------
        task_name : string
            the name of the classification task to be imported
        is_read_feature_data : bool
            if true, the feature data matrix will be read (default is True)
        Returns:
        --------
        data : ndarray
            Array containing the data. Each row corresponds to a timeseries and each column to an operation.
        ts : dict
            dictionary containing the information for all timeseries (rows in data). 
            ['keywords', 'n_samples', 'id', 'filename']
        op : dict
            dictionary containing the information for all contained operations. 
            Keys are ['keywords', 'master_id', 'id', 'code_string', 'name']
            
        """
        # -- assemble the file path
        mat_file_path = self.path_pattern.format(task_name)
        # -- load the data,operations and timeseries from the matlab file
        if is_read_feature_data:
            data , op, ts = mIO.read_from_mat_file(mat_file_path,['TS_DataMat','Operations','TimeSeries'],is_from_old_matlab = True)
        else:
            op, ts = mIO.read_from_mat_file(mat_file_path,['Operations','TimeSeries'],is_from_old_matlab = True)
            data = None
        
        if is_read_feature_data:
            return self.masking_method(data), ts, op
        else:
            return None, ts, op
