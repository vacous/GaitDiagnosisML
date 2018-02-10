import numpy as np
from sklearn.decomposition import PCA
from scipy import signal
import bisect

class RescalePCA:
    def __init__(self):
        '''data has fields [f1, f2, ..., Time]'''
        self.PCAProcess = None
        self.scale_info = {}
        self.component_range = None
        self.feature_order = {}
        
    def __str__(self):
        pca_info = 'No data feed for PCA'
        scale_info = 'No data feed for Rescale'
        component_range = 'No range calculated'
        feature_order = 'No features are given'
        if self.PCAProcess is not None: 
            pca_info = 'PCA Info: \n'
            pca_info += str(self.PCAProcess.explained_variance_ratio_)
        if len(self.scale_info) != 0: 
            scale_info = 'Rescale Info:'
            for f in self.scale_info:
                scale_info += '\n' + f + ': ' + str(self.scale_info[f]) 
        if self.component_range is not None:
            comp_str = 'Component Range: \n'
            comp_str += str(self.component_range[0]) + '\n' + str(self.component_range[1])
        if len(self.feature_order) != 0:
            fea_str = 'Feature Order: '
            fea_str += str(self.feature_order)
        return pca_info + '\n' + scale_info + '\n' + comp_str + '\n' + fea_str
    
    def saveFeatureOrder(self, data_frame):
        cur_idx = 0
        for fea in list(data_frame):
            if fea != 'Time':
                self.feature_order[fea] = cur_idx
                cur_idx += 1

    def getRescaleInfo(self, data_frame):        
        for field in list(data_frame):
            if field != 'Time':
                cur_data = data_frame[field]
                cur_max, cur_min = np.max(cur_data), np.min(cur_data)
                self.scale_info[field] = (cur_max, cur_min)
    
    def applyRescale(self, data_frame):
        out_df = data_frame.copy()
        for fea in list(out_df):
            if fea != 'Time': 
                out_df[fea] = (out_df[fea] - self.scale_info[fea][1])/(self.scale_info[fea][0] - self.scale_info[fea][1])
        return out_df
        
    def processRescalePCA(self, all_data, reduced_dim = 3):
        '''Rescale the data, and then apply the PCA reduction'''
        self.saveFeatureOrder(all_data)
        self.getRescaleInfo(all_data)
        all_rescaled_data = self.applyRescale(all_data)
        # Fit data into PCA to obtain the dimension reduction information
        reorder_data = self.reorderData(all_rescaled_data)
        self.PCAProcess = PCA(n_components= reduced_dim)
        self.PCAProcess.fit(reorder_data)
        # Transform the original data to obtain the range for each reduced dimension
        reduced_data_mat = self.PCAProcess.transform(reorder_data)
        self.component_range = (np.min(reduced_data_mat,0), np.max(reduced_data_mat,0))
        self.roundSecond(self.component_range)
        
    def applyRescalePCA(self, other_data):
        '''Use the pre computed sacle and PCA info to process new data'''
        rescaled_data = self.applyRescale(other_data)
        reorder_rescaled_data = self.reorderData(rescaled_data)
        PCA_data = self.PCAProcess.transform(reorder_rescaled_data)
        return PCA_data

    # helpers 
    def reorderData(self, some_data):
        '''
        make sure the data frame organized with the same order as the initial input
        also remove the 'Time' column, so that the dataframe can be used for PCA
        '''
        data_order = [None for _ in range(len(self.feature_order))]
        for fea in self.feature_order:
            data_order[self.feature_order[fea]] = fea
        return some_data[data_order]
        
    def roundSecond(self, old_range):
        '''
        old_range = (np.array(mins), np.array(maxs))
        round to 2nd place 
        '''
        for each in old_range:
            for idx in range(len(each)):
                if each[idx] > 0: each[idx] = np.ceil(each[idx]*100)/100
                else: each[idx] = np.floor(each[idx]*100)/100

                
# Extract Features from RescalePCA processed Data,
# after having the RescalePCA infomation 
def HistFeature(in_data, all_range, fea_num = 5):
    num_fields = in_data.shape[1]
    out_features = np.zeros((fea_num * num_fields))
    for idx in range(in_data.shape[1]):
        cur_col = in_data[:,idx]
        cur_hist = np.histogram(cur_col, range = (all_range[0][idx], all_range[1][idx]), bins = fea_num)[0]
        out_features[idx * fea_num: (idx + 1) * fea_num] = cur_hist/np.sum(cur_hist)
    return out_features

def InterpolateHistCount(count_range, xs, ys, bins):
    ''' count range = (min, max) '''
    out_bins = np.zeros(bins)
    sep_vals = np.linspace(count_range[0], count_range[1], bins)
    sep_dist = (count_range[1] - count_range[0])/(bins-1)
    for idx in range(len(xs)):
        x = xs[idx]
        y = ys[idx]
        # interpolate count 
        upper_idx = bisect.bisect(sep_vals, x)
        lower_idx = upper_idx - 1
        upper_dist = sep_vals[upper_idx] - x
        lower_dist = x - sep_vals[lower_idx]
        # add count to the count bins 
        out_bins[upper_idx] += lower_dist/sep_dist * y**2
        out_bins[lower_idx] += upper_dist/sep_dist * y**2
    # normalize the out_bins so that the sum is still one 
    return out_bins/np.sum(out_bins)

def FFTPeaks(cur_data, in_time, cut_off_sig_len, num_peaks):
    sample_freq = np.mean(1/(np.diff(in_time)))
    sample_period = 1/sample_freq
    sig_len = len(cur_data)
    fft_result = np.fft.fft(cur_data)
    sig_fft = np.abs(fft_result/sig_len)
    # plot fft freq result 
    hf_len = int(sig_len/2) 
    fft_y = sig_fft[1:hf_len + 1]
    fft_y[1:-1] = 2 * fft_y[1:-1]
    freq_x = sample_freq * range(0,hf_len)/sig_len
    # find peaks and generate features with interpolative hist count 
    all_maxs_idxs = signal.find_peaks_cwt(fft_y[:cut_off_sig_len], np.arange(0.1, 1, 0.2), noise_perc = 99)
    maxs_idxs = [each[1] for each in sorted([(fft_y[idx], idx) for idx in all_maxs_idxs], reverse=True)[:num_peaks]]
    return freq_x,fft_y,maxs_idxs

def FFTFeature(pca_data, in_time, cut_off_sig_len = 250, fea_num = 10, num_peaks = 5):
    out_fft_features = np.zeros(pca_data.shape[1] * fea_num)
    for idx in range(pca_data.shape[1]):
        cur_data = pca_data[:,idx]
        freq_x, fft_y, maxs_idxs = FFTPeaks(cur_data, in_time, cut_off_sig_len, num_peaks)
        x_upper_limit = freq_x[cut_off_sig_len]
        max_freqs = freq_x[maxs_idxs]
        max_fft_y = fft_y[maxs_idxs]/np.sum(fft_y[maxs_idxs])
        cur_fft_features = InterpolateHistCount([0, x_upper_limit], max_freqs, max_fft_y, fea_num)
        out_fft_features[idx * fea_num: (idx + 1) * fea_num] = cur_fft_features
    return out_fft_features