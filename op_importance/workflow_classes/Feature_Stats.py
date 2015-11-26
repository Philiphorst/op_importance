import modules.feature_importance.PK_test_stats as fistat
import numpy as np

class Feature_Stats:
    
    def __init__(self,combine_pair_method = 'mean'):
        """
        Constructor
        Parameters:
        -----------
        combine_pair_method : str
            String describing the method used to combine the pairwise calculated statistics (if applicable)
        Returns:
        --------
        None
        """
        if combine_pair_method == 'mean':
            self.combine_pair = self.combine_pair_stats_mean
        else:
            self.combine_pair = self.combine_pair_not
   
    def combine_pair_not(self,pair_stats):
        """
        Pairs are not combined either because the statistics used does not require combination or because combination 
        is not desires.
        Parameters:
        -----------
        pair_stats : ndarray
            Array containing the stats for each label pair (row) for all operations (columns)
        Returns:
        --------
        pair_stats: ndarray
            Unchanged input
        """       
        return pair_stats
    
    
    def combine_pair_stats_mean(self,pair_stats):
        """
        Combine the stats of the label pairs to one pooled stat by taking the average along the columns of
        pair_stats (operations)
        Parameters:
        -----------
        pair_stats : ndarray
            Array containing the stats for each label pair (row) for all operations (columns)
        Returns:
        --------
        combined_stats:
            Average over all label pairs for each feature
        """
        # -- combine the stas for all labels pairs by taking the mean
        return np.ma.average(pair_stats,axis=0)


class U_Stats(Feature_Stats):
    def __init__(self,combine_pair_method = 'mean'):        
        """
        Constructor
        Parameters:
        -----------
        combine_pair_method : str
            String describing the method used to combine the pairwise calculated statistics (if applicable)
        Returns:
        --------
        None
        """
        Feature_Stats.__init__(self,combine_pair_method = 'mean')
    
    def calc_pairs(self,labels,data):
        """
        Calculate the ustatistic for each operation and every label pairing
        Parameters:
        -----------
        labels : ndarray
            1-D array containing the labels for each row in data.
        data : ndarray
            Array containing the data. Each row corresponds to a timeseries and each column to an operation.
        Returns:
        --------
        ranks : ndarray
            Returns the scaled U statistic for each label pairing and each operation.
    
        """
        ranks,ustat_norm = fistat.u_stat_all_label(data,labels=labels)[0:2]
        return ranks/ustat_norm[:,np.newaxis]
