import modules.misc.PK_matlab_IO as mIO
import re
import numpy as np
import modules.misc.SQL_IO as sIO
import numpy.ma as ma


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
        keywords_ts : list
            list containing the keywords for all timeseries (rows in data)
        op_ids : ndarray
            1-D array containing the operation id for each column in data
            
        """
        # -- assemble the file path
        mat_file_path = self.path_pattern.format(task_name)
        # -- load the data,operations and timeseries from the matlab file
        if is_read_feature_data:
            data , op, ts = mIO.read_from_mat_file(mat_file_path,['TS_DataMat','Operations','TimeSeries'],is_from_old_matlab = True)
        else:
            op, ts = mIO.read_from_mat_file(mat_file_path,['Operations','TimeSeries'],is_from_old_matlab = True)
            data = None
        # -- extract the op ids for the columns 
        op_ids = np.array(op['id'])
        keywords_ts = ts['keywords']
        if is_read_feature_data:
            return self.masking_method(data), keywords_ts, op_ids
        else:
            return None, keywords_ts, op_ids
        

class Database_Input(Data_Input):
    #FIXME masking_method is a dummy argument, this might seem confusing 
    def __init__(self, masking_method = 'nan',label_regex_pattern = '.*,(.*)$'):
        """
        Constructor
        Parameters:
        -----------
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
        keywords_ts : list
            list containing the keywords for all timeseries (rows in data)
        op_ids : ndarray
            1-D array containing the operation id for each column in data
            
        """
        if is_read_feature_data:
            data , op, ts = sIO.read_from_sql_by_filename(task_name,['TS_DataMat','Operations','TimeSeries'])
        else:
            op, ts = sIO.read_from_sql_by_filename(task_name,['Operations','TimeSeries'])
            data = None
        # -- extract the op ids for the columns 
        op_ids = np.array(op['id'])
        keywords_ts = ts['keywords']
        if is_read_feature_data:
            return ma.getdata(data), keywords_ts, op_ids
        else:
            return None, keywords_ts, op_ids
        
if __name__ == '__main__':
    
    path_pattern = '/home/bjm113/Downloads/HCTSA_{:s}_N_70_100_reduced.mat'
    
    masking_method = 'NaN'
    label_regex_pattern = '.*,(.*)$'
    
    input_method = Database_Input(path_pattern)
    matlab_input_method = Datafile_Input(path_pattern)

    print input_method.input_task('Coffee_%') 
    print matlab_input_method.input_task('Lighting2')
