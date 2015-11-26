import numpy as np
import matplotlib.pyplot as plt
import modules.feature_importance.PK_ident_top_op as idtop
import scipy.cluster.hierarchy as hierarchy
import modules.misc.PK_helper as hlp
import modules.misc.PK_matlab_IO as mIO
import re

intermediate_data_root = '../data/'
all_classes_avg_out_path = intermediate_data_root+'/all_classes_avg.npy'
op_id_order_path = intermediate_data_root +'/op_id_order.npy'
op_id_good_path = intermediate_data_root +'/op_id_good.npy'
problem_names_path = intermediate_data_root+'problem_names.npy'
measures_problems_path = intermediate_data_root+'measure_problems.npy'
# -- Load the data
all_classes_avg = np.load(all_classes_avg_out_path)
op_id_order = np.load(op_id_order_path)
op_id_good = np.load(op_id_good_path)
max_feat = 50
max_corr_dist = 0.2
# # -- mask all nan values and take top 200 features
# all_classes_avg_top = np.ma.masked_invalid(all_classes_avg[:,op_id_order[:100]])
# # -- calculate the z-score of the u stat array
# all_classes_avg_top = ((all_classes_avg_top.T - np.ma.mean(all_classes_avg_top,axis=1)) / np.ma.std(all_classes_avg_top,axis=1)).T
# abs_corr_array = np.abs(np.ma.corrcoef(all_classes_avg_top, rowvar=0)) 

# -- calculate the correlation array with respect to performance and mask nan.
abs_corr_array,sort_good_ind,all_classes_avg_good_norm = idtop.calc_perform_corr_mat(all_classes_avg[:,op_id_good],norm='z-score', max_feat = max_feat)
all_classes_avg_top = np.ma.masked_invalid(all_classes_avg[:,op_id_good][:,sort_good_ind[:max_feat]])

# -- calculate the linkage for the correlation
corr_linkage = idtop.calc_linkage(abs_corr_array)[0]

# -- extract operation names --- ------------------------------------------
# -- load a reference HCTSA_loc.mat containing all op_ids
op_ref_HCTSA_path = '/home/philip/work/OperationImportanceProject/results/done/HCTSA_Beef.mat'
op, = mIO.read_from_mat_file(op_ref_HCTSA_path,['Operations'],is_from_old_matlab = True)   

top_id = op_id_good[sort_good_ind][:max_feat]
names = hlp.ind_map_subset(op['id'], op['name'], op_id_good[sort_good_ind][:max_feat])

# -- extract problem names --- ------------------------------------------
reg_ex = re.compile('.*\/HCTSA_(.*)_N_70_100_reduced.mat')
problem_paths = np.load(problem_names_path)

problem_names = np.array([reg_ex.match(problem_path).group(1) for problem_path in problem_paths])


# ---------------------------------------------------------------------
# -- Plot -------------------------------------------------------------
# ---------------------------------------------------------------------

fig = plt.figure(figsize = ((15,15)))
# -- plot layout ------------------------------------------------------

rect_ustat_arr = [0.25,0.275,.6,.6]
rect_dendr = [0.855,0.275,.135,.6]
rect_measures = [0.25,0.88,0.6,0.1]

rect_color = [0.25,0.08,0.6,0.02]

ax_ustat_arr = fig.add_axes(rect_ustat_arr)
ax_dendr = fig.add_axes(rect_dendr)
ax_colorbar = fig.add_axes(rect_color)
ax_measures = fig.add_axes(rect_measures)
ax_measures1 = plt.twinx(ax_measures) 

# -- plot dendrogram --------------------------------------------------
corr_dendrogram = hierarchy.dendrogram(corr_linkage, orientation='left',no_plot=True)
hierarchy.dendrogram(corr_linkage, orientation='left',p=50,truncate_mode ='lastp',ax = ax_dendr)
ax_dendr.set_yticks([])

ax_dendr.axvline(max_corr_dist,ls='--',c='k')

# -- plot sorted U-Stat array ------------------------------------------

# -- create index that sort rows to correspond to dendrogram
feat_sort_ind = corr_dendrogram['leaves']
# -- create index that sort columns with respect to their mean value
porblem_sort_ind = np.argsort(all_classes_avg_top[:,feat_sort_ind].mean(axis=1))
print porblem_sort_ind
#all_classes_avg_top = ((all_classes_avg_top - np.ma.mean(all_classes_avg_top,axis=0)) / np.ma.std(all_classes_avg_top,axis=0))

# -- plot the operation names as y-axis tick labels
im_ustat_arr = ax_ustat_arr.matshow(all_classes_avg_top[porblem_sort_ind,:][:,feat_sort_ind].T,aspect=39/float(50),origin='bottom')
ax_ustat_arr.set_yticks(range(len(feat_sort_ind)))
ax_ustat_arr.set_yticklabels(np.array(names)[feat_sort_ind])
# -- plot the problem names as x axis labels
ax_ustat_arr.xaxis.tick_bottom()
ax_ustat_arr.set_xticks(range(all_classes_avg_top.shape[0]))
ax_ustat_arr.set_xticklabels(problem_names[porblem_sort_ind],rotation='vertical')

# -- calculate and plot clusters ----------------------------------
cluster_ind = hierarchy.fcluster(corr_linkage, t = max_corr_dist, criterion='distance')
cluster_bounds = np.nonzero(np.diff(cluster_ind[feat_sort_ind]))[0]+0.5
for cluster_bound in cluster_bounds:
    ax_ustat_arr.axhline(cluster_bound,c='w',lw=2)

# -- calculate and plot measures
if False:
    measures = np.zeros((2,problem_paths.shape[0]))
    for i,mat_file_path in enumerate(problem_paths):
        ts, = mIO.read_from_mat_file(mat_file_path,['TimeSeries'],is_from_old_matlab = True)   
        print i
        measures[0,i] = len(list(set([int(kw.split(',')[-1])for kw in ts['keywords']])))
        measures[1,i] = np.mean([int(samples) for samples in ts['n_samples']])
        print measures[:,i]
    np.save(measures_problems_path, measures)
else:
    measures = np.load(measures_problems_path)  


x_loc = np.arange(0,problem_paths.shape[0])+0.5
ax_measures.scatter(x_loc,measures[0,porblem_sort_ind],c='b',s=40)
ax_measures.plot(x_loc,measures[0,porblem_sort_ind],c='b')

[label.set_color('b') for label in ax_measures.get_yticklabels()]
ax_measures.set_ylabel('nr classes')
ax_measures.yaxis.label.set_color('b')
ax_measures.set_ylim([0,80])
ax_measures.set_xlim([0,problem_paths.shape[0]])
ax_measures.set_xticklabels([])

ax_measures1.scatter(x_loc,measures[1,porblem_sort_ind],c='r',s=40)
ax_measures1.plot(x_loc,measures[1,porblem_sort_ind],c='r')

[label.set_color('r') for label in ax_measures1.get_yticklabels()]
ax_measures1.set_ylabel('avg samples')
ax_measures1.yaxis.label.set_color('r')
ax_measures1.set_ylim([0,2000])


ax_measures.set_xlim([0,problem_paths.shape[0]])
ax_measures.set_xticklabels([])

# -- plot colorbar  ----------------------------------
cb = plt.colorbar(mappable = im_ustat_arr, cax=ax_colorbar, orientation = 'horizontal') 
cb.set_label('Average U-Statistics for each problem')
# all_classes_avg_masked_sort =  np.ma.masked_invalid(all_classes_avg[porblem_sort_ind,:][:,op_id_good])
# print (~all_classes_avg_masked_sort.mask).sum(axis=1)
# ax_measures1.plot(x_loc,(~all_classes_avg_masked_sort.mask).sum(axis=1))
plt.savefig('/home/philip/Desktop/tmp/figure_tmp/u_stat_array_sngl.png')

plt.show()






