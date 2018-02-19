import matplotlib.pyplot as plt
from FeatureAnalysis import FFTPeaks

def VisMeasurements(record, plt_len = 3000):
    mea_types = {}
    for each_fea in list(record):
        if each_fea != 'Time':
            main_type = each_fea[:each_fea.index('_')]
            if main_type not in mea_types: mea_types[main_type] = [each_fea]
            else: mea_types[main_type].append(each_fea)
    # set plot range 
    time_record = record['Time']
    plt_end = plt_len if plt_len < len(time_record) else -1
    # plot for each measurement type
    for each_mea_type in mea_types:
        plt.figure(figsize=(15,5))
        cur_mea_names = mea_types[each_mea_type]
        for idx in range(len(cur_mea_names)):
            plt.subplot(1, len(cur_mea_names), idx + 1)
            plt.title(cur_mea_names[idx])
            plt.plot(time_record[:plt_end], record[cur_mea_names[idx]][:plt_end])
            plt.xlabel('Time/sec')
            plt.ylabel(cur_mea_names[idx])
        plt.tight_layout()
        plt.suptitle(each_mea_type, fontsize = 15)
        plt.subplots_adjust(top=0.8)

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

def VisFFTResult(pca_data, in_time, cut_off_sig_len, num_peaks = 3, data_name = ""):
    num_subs = pca_data.shape[1]
    plt.figure(figsize=(15,5))
    for idx in range(num_subs):
        cur_data = pca_data[:,idx]
        freq_x, fft_y, max_idxs = FFTPeaks(cur_data, in_time, cut_off_sig_len, num_peaks)
        # plot the original result 
        plt.subplot(1,num_subs, idx + 1)
        plt.title('Component: ' + str(idx + 1))
        plt.plot(freq_x[:cut_off_sig_len], fft_y[:cut_off_sig_len])
        plt.plot(freq_x[max_idxs], fft_y[max_idxs],'ro')
        max_idxs = set(max_idxs)
        fft_y[[idx for idx in range(cut_off_sig_len) if idx not in max_idxs]] = 0
        # plot the processed results 
        plt.plot(freq_x[:cut_off_sig_len], fft_y[:cut_off_sig_len], ":")
        plt.xlabel('Frequency')
        plt.ylabel('Magnitude')
    data_info = ": " if len(data_name) != 0 else "" 
    data_info += data_name
    plt.suptitle('FFT Analysis Result' + data_info, fontsize = 15)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)

def VisTrainVsTest(train_matrix, train_label, gait_types,
                                test_case, test_case_name,
                                vis_feature_trans = 0.07, num_same_type = 300):
    all_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt_starts = {}
    for idx in range(len(train_label)):
        cur_gait_name = gait_types[int(train_label[idx])]
        if cur_gait_name not in plt_starts: plt_starts[cur_gait_name] = idx
    for idx in range(len(gait_types)):
        each_type = gait_types[idx]
        plt_range = range(plt_starts[each_type], plt_starts[each_type] + num_same_type)
        plt.figure(figsize=(15,5))
        for plt_idx in plt_range:
            plt.plot(train_matrix[plt_idx,:], all_colors[idx], alpha=vis_feature_trans)
        plt.plot(train_matrix[plt_idx,:], all_colors[idx], label = each_type)
        plt.plot(test_case, 'xkcd:gold', linewidth=3.0, label = test_case_name)
        plt.legend()
        plt.ylim(-0.1,1)

def VisPredResult(pred_prob, label_list, test_case_name = ''):
    plt.figure(figsize=(8,5))
    temp_y = range(len(label_list))
    plt.barh(temp_y, pred_prob, ec = 'black', alpha = 0.8)
    plt.yticks(temp_y, label_list)
    plt.xlabel('Predicted Probability', fontsize = 13)
    plt.ylabel('Categories', fontsize = 13)
    plt.title('Classification Result' + ((' ' if len(test_case_name) != 0 else '') + test_case_name))
    plt.grid()
    plt.tight_layout()