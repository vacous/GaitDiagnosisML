import matplotlib.pyplot as plt

def VisCompareTwoPCA(pca_data_1, pca_data_2, PCA_process_obj, data_name_1, data_name_2):
    '''two compressed pca_data, the object used to process the data, two data names for plot'''
    fig = plt.figure(figsize=(15,10))
    for plt_idx in range(pca_data_1.shape[1]):
        cur_plt_range = PCA_process_obj.component_range
        plt.subplot(3,2,2*plt_idx+1)
        plt.title(data_name_1 + ': ' + str(plt_idx))
        plt.hist(pca_data_1[:,plt_idx], alpha=0.5, ec = 'black', range= [cur_plt_range[0][plt_idx], cur_plt_range[1][plt_idx]] )
        plt.subplot(3,2, 2*plt_idx+2)
        plt.title(data_name_2 + ': ' + str(plt_idx))
        plt.hist(pca_data_2[:,plt_idx], alpha = 0.5, ec = 'black', range= [cur_plt_range[0][plt_idx], cur_plt_range[1][plt_idx]])
    plt.suptitle('Comparison between ' + data_name_1 + ' and ' +  data_name_2, fontsize = 15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)