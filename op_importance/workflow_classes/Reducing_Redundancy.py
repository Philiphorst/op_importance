import modules.misc.PK_helper as hlp
import modules.feature_importance.PK_ident_top_op as idtop
import scipy.cluster.hierarchy as hierarchy

class Reducing_Redundancy:
    def __init__(self,similarity_method,compare_space):
        """
        Constructor
        Parameters:
        -----------
        similarity_method : string
            String secribing the method used to calculate the distance array
        compare_space : string
            String describing in which space the distance calculation is going to appen (e.g. problem_stats,feature_vals)
        """
        if similarity_method == 'correlation':
            self.calc_similarity = self.calc_abs_corr
            
        self.compare_space = compare_space   
        self.good_perf_op_ids = None
        self.ops_base_perf_vals = None
        self.similarity_array = None
        self.cluster_inds = None # Indices to which cluster an entry of the similarity array belongs
        self.cluster_op_id_list= None # List of lists containing the op_id for the operations in the clusters
    def set_parameters(self,ops_base_vals,good_op_ids,good_perf_op_ids):
        """
        Set and compute the parameters needed to calculate the distance array
        Parameters:
        -----------
        ops_base_vals : nd array
            Array containing the values on which the similarity of the operations will be calculated
        good_op_ids : ndarray
            The op_ids of the columns in  ops_base_vals
        good_perf_op_ids : ndarray
            The op_ids of the features we are interested in
        """
        self.good_perf_op_ids = good_perf_op_ids
        self.good_op_ids = good_op_ids
        # -- This discards the potential large ops_base_vals and also good_op_ids after exiting the constructor
        self.ops_base_perf_vals = self.reduce_to_good_perf_ops(ops_base_vals,self.good_perf_op_ids,good_op_ids)
        

    def reduce_to_good_perf_ops(self,ops_base_vals,good_perf_op_ids,good_op_ids):
        """
        Reduce the ops_base_vals by keeping only the columns corresponding to the op_ids in self.good_perf_op_ids
        Parameters:
        -----------
        ops_base_vals : nd array
            Array containing the values on which the similarity of the operations will be calculated
        good_op_ids : ndarray
            The op_ids of the columns in  ops_base_vals
        good_perf_op_ids : ndarray
            The op_ids of the features we are interested in
        Returns:
        --------
        ops_base_perf_vals : ndarray
            ops_base_vals reduced to contain only operations with ids given in good_perf_op_ids with the same orgering.
        """
        good_perf_ind = hlp.ismember(good_perf_op_ids,good_op_ids)
        ops_base_perf_vals = ops_base_vals[:,good_perf_ind]
        return ops_base_perf_vals
    
    def calc_abs_corr(self):
        """
        Calculate the distance matrix using a correlation approach for every column in self.ops_base_perf_vals
        """
        # -- no normalisation in here as the best performing features have been picked already, potentially using normalisation

        self.similarity_array,_,_ = idtop.calc_perform_corr_mat(self.ops_base_perf_vals,norm=None, 
                                                              max_feat = self.ops_base_perf_vals.shape[1])
        
    def calc_hierch_cluster(self,t = 0.2, criterion='distance' ):
        """
        Calculate the clustering using hierachical clustering
        Parameters:
        -----------
        t : float
            The threshold to apply when forming flat clusters.
        criterion : str, optional
            The criterion to use in forming flat clusters. 
        """
        corr_linkage = idtop.calc_linkage(self.similarity_array)[0]
        self.cluster_inds = hierarchy.fcluster(corr_linkage, t = t, criterion=criterion)
        # -- map index to op_id and create list of lists representing clusters
        self.cluster_op_id_list = [[] for x in xrange(self.cluster_inds.max())]
        for i,cluster_ind in enumerate(self.cluster_inds):
            self.cluster_op_id_list[cluster_ind-1].append(self.good_perf_op_ids[i])

        
class Correlation_Dist:
    def __init__(self):
        pass