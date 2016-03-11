import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hierarchy

import modules.feature_importance.PK_feat_array_proc as fap
import modules.misc.PK_helper as hlp
import modules.feature_importance.PK_plot_functions as fiplt

import modules.feature_importance.PK_ident_top_op as idtop

class Plotting:
    def __init__(self,workflow,max_dist_cluster = 0.2):
        """Constructor
        Parameters:
        -----------
        workflow: Workflow instance
            Instance of the Workflow class providing access to the calculated and imported values
        max_dist_cluster : float
            The maximum distance in the similarity array for each cluster
        """
        self.workflow = workflow
        self.linkage = workflow.redundancy_method.linkage
        self.max_dist_cluster = max_dist_cluster
        self.ops_base_perf_vals = workflow.redundancy_method.ops_base_perf_vals
        
        
        
        self.task_names = np.array([task.name for task in workflow.tasks])
    
    
    def map_op_id_name_mult_task(self,tasks):  
        """
        Combines all operation id's from all tasks, removes duplicates and maps them to their respective op names
        Parameters:
        -----------
        tasks : list of instances of Task class
            The list containing all considered tasks with the required class variables set (op['id'] and op['name'])
        Returns:
        --------
        retval : list of lists
            A list of lists where the first sub-list corresponds to the operation id and the second sub-list to the operation name.
        """
        return zip(*set(list(itertools.chain.from_iterable([zip(task.op['id'],task.op['name']) for task in tasks]))))
    
    def plot_similarity_array(self):
        
        abs_corr_array = self.workflow.redundancy_method.similarity_array
        
        op_id_name_map = self.map_op_id_name_mult_task(self.workflow.tasks)
        names = hlp.ind_map_subset(op_id_name_map[0],op_id_name_map[1],self.workflow.redundancy_method.similarity_array_op_ids)
        measures = np.zeros((2,len(names)))
       
        
        tmp_ind = hlp.ismember(self.workflow.redundancy_method.similarity_array_op_ids, 
                       self.workflow.good_op_ids)
        
        # -- number of problems for which each good performing feature has been calculated
        measures[0,:] = (~self.workflow.stats_good_op[:,tmp_ind].mask).sum(axis=0)
        # -- z scored u-stat(for all features) for top features 
        stats_good_op_z_score = fap.normalise_masked_array(self.workflow.stats_good_op_comb, axis= 0,norm_type = 'zscore')[0]
        measures[1,:] = stats_good_op_z_score[tmp_ind]
        
        fiplt.plot_arr_dendrogram(abs_corr_array,names,max_dist_cluster=self.max_dist_cluster,measures = measures)
        
        #abs_corr_array,sort_good_ind,all_classes_avg_good_norm = idtop.calc_perform_corr_mat(self.workflow.stats_good_op,norm='z-score')
        
    def plot_stat_array(self):
        
        fig = plt.figure(figsize = ((15,15)))
        # -- plot layout ------------------------------------------------------
        
        rect_ustat_arr = [0.25,0.175,.5,.5]
        rect_dendr = [0.755,0.175,.145,.5]
        rect_measures0 = [0.25,0.68,0.5,0.1]
        rect_measures1 = [0.25,0.785,0.5,0.1]
        rect_measures2 = [0.25,0.89,0.5,0.1]
        
        ax_ustat_arr = fig.add_axes(rect_ustat_arr)
        ax_dendr = fig.add_axes(rect_dendr)
        ax_measures00 = fig.add_axes(rect_measures0)
        ax_measures01 = plt.twinx(ax_measures00) 
        ax_measures10 = fig.add_axes(rect_measures1)
        ax_measures10.set_xticklabels([])
        ax_measures20 = fig.add_axes(rect_measures2)
        
        ax_measures20.set_xticklabels([])
        ax_measures21 = plt.twinx(ax_measures20)
        
        # -- calculate and plot the dendrogram
        dist_dendrogram = hierarchy.dendrogram(self.linkage, orientation='left',no_plot=True)
        hierarchy.dendrogram(self.linkage, orientation='left',p=50,truncate_mode ='lastp',ax = ax_dendr)
        
        ax_dendr.set_yticks([])
        ax_dendr.axvline(self.max_dist_cluster,ls='--',c='k')

        # -- plot sorted U-Stat array ------------------------------------------

        # -- create index that sort rows to correspond to dendrogram
        feat_sort_ind = dist_dendrogram['leaves']
        # -- sort the good performant features so they have the same order as the similarity array
        sort_ind = hlp.ismember(self.workflow.redundancy_method.similarity_array_op_ids,self.workflow.redundancy_method.good_perf_op_ids)
        self.ops_base_perf_vals = self.ops_base_perf_vals[:,sort_ind]
        
        # -- create index that sort columns with respect to their mean value
        task_sort_ind = np.argsort(self.ops_base_perf_vals[:,feat_sort_ind].mean(axis=1))
        
        #all_classes_avg_top = ((all_classes_avg_top - np.ma.mean(all_classes_avg_top,axis=0)) / np.ma.std(all_classes_avg_top,axis=0))
        #all_classes_avg_top = fap.normalise_masked_array(self.ops_base_perf_vals, axis= 1,norm_type = 'zscore')[0]
        all_classes_avg_top = self.ops_base_perf_vals
        # -- plot the operation names as y-axis tick labels
        aspect = all_classes_avg_top.shape[0] / float(all_classes_avg_top.shape[1])
        ax_ustat_arr.matshow(all_classes_avg_top[task_sort_ind,:][:,feat_sort_ind].T,aspect=aspect,origin='bottom')


        ax_ustat_arr.set_yticks(range(len(feat_sort_ind)))

        op_id_name_map = self.map_op_id_name_mult_task(self.workflow.tasks)
        names = hlp.ind_map_subset(op_id_name_map[0],op_id_name_map[1],self.workflow.redundancy_method.similarity_array_op_ids)
        ax_ustat_arr.set_yticklabels(np.array(names)[feat_sort_ind])
        # -- plot the problem names as x axis labels
        ax_ustat_arr.xaxis.tick_bottom()
        ax_ustat_arr.set_xticks(range(all_classes_avg_top.shape[0]))
        ax_ustat_arr.set_xticklabels(self.task_names[task_sort_ind],rotation='vertical')

        # -- plot clusters ----------------------------------
        
        cluster_bounds = np.nonzero(np.diff(self.workflow.redundancy_method.cluster_inds[feat_sort_ind]))[0]+0.5
        for cluster_bound in cluster_bounds:
            ax_ustat_arr.axhline(cluster_bound,c='w',lw=2)
    
        # --------------------------------------------------------------------------------
        # -- calculate and plot measures -------------------------------------------------
        # --------------------------------------------------------------------------------

        # -- nr samples and nr labels --------------------------------------------------
  
        n_samples_avg = [np.array(self.workflow.tasks[i].ts['n_samples']).mean() for i in task_sort_ind]
        n_classes = [len(set(self.workflow.tasks[i].labels)) for i in task_sort_ind]
        x_loc = np.arange(0,len(self.workflow.tasks))+0.5
        ax_measures00.scatter(x_loc,n_classes,c='b',s=40)
        ax_measures00.plot(x_loc,n_classes,c='b')
        [label.set_color('b') for label in ax_measures00.get_yticklabels()]
        ax_measures00.set_ylabel('nr classes')
        ax_measures00.yaxis.label.set_color('b')
        ax_measures00.set_ylim([0,max(n_classes)+1])
        ax_measures00.set_xticklabels([])
    
        ax_measures01.scatter(x_loc,n_samples_avg,c='r',s=40)
        ax_measures01.plot(x_loc,n_samples_avg,c='r')
        
        [label.set_color('r') for label in ax_measures01.get_yticklabels()]
        ax_measures01.set_ylabel('avg samples')
        ax_measures01.yaxis.label.set_color('r')
        ax_measures01.set_ylim([0,max(n_samples_avg)+100])
        
        ax_measures00.set_xlim([0,len(self.workflow.tasks)])
        ax_measures00.set_xticklabels([])
        

        # -- U-stat measures --------------------------------------------------
        
        # -- minimum average U-score for all features
        ax_measures10.plot(x_loc,np.min(self.workflow.stats_good_op[task_sort_ind,:],axis=1),marker='o',label='min. avg. U-score all')

        # -- minimum average U-score for top features
        ax_measures10.plot(x_loc,np.ma.min(self.ops_base_perf_vals[task_sort_ind,:],axis=1),marker='o',label='min. avg. U-score top')
            
        # -- average minimum (for each class pair) U-score for top features
        # XXX This would require task.pair_stats to be available (not saved as intermediate at the moment); then it is trivial to implement
        
        ax_measures10.legend(loc=2,fontsize='small',labelspacing=.1)
        ax_measures10.set_ylabel('u-score')
        ax_measures10.set_xlim([0,len(self.workflow.tasks)])
        ax_measures10.set_ylim([0,0.5])

        # -- U-stat measures and avg operations working--------------------------------------------------
        # -- mean average U-score for all features
        ax_measures20.plot(x_loc,np.ma.mean(self.workflow.stats_good_op[task_sort_ind,:],axis=1),marker='o')
        [label.set_color('b') for label in ax_measures20.get_yticklabels()]
        ax_measures20.set_ylabel('avrg u-scrore all feat')
        ax_measures20.yaxis.label.set_color('b')
        
        # -- number of successfully calculated features

        ax_measures21.plot(x_loc,[len(self.workflow.tasks[i].op['id']) for i in task_sort_ind],c='r',marker='o')
        [label.set_color('r') for label in ax_measures21.get_yticklabels()]
        ax_measures21.set_ylabel('nr calc feat')
        ax_measures21.yaxis.label.set_color('r')
        
        ax_measures20.set_xlim([0,len(self.workflow.tasks)])
        
