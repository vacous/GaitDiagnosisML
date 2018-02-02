import numpy as np
from sklearn.decomposition import PCA

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
            fea_str += feature_order
        return pca_info + '\n' + scale_info + '\n' + comp_str + '\n' + fea_str
    
    def saveFeatureOrder(self, xs_dict):
        cur_idx = 0
        for fea in xs_dict:
            if fea != 'Time':
                self.feature_order[fea] = cur_idx
                cur_idx += 1
        self.feature_order['Time'] = cur_idx

    def getRescaleInfo(self, xs_dict):
        for field in xs_dict:
            if field != 'Time':
                cur_data = xs_dict[field]
                cur_max, cur_min = np.max(cur_data), np.min(cur_data)
                self.scale_info[field] = (cur_max, cur_min)
    
    def applyRescale(self, xs_dict):
        out_dict = {}
        for f in xs_dict:
            if f != 'Time':
                cur_data = xs_dict[f]
                cur_max, cur_min = self.scale_info[f][0], self.scale_info[f][1]
                out_dict[f] = (cur_data - cur_min)/(cur_max - cur_min)
            else: out_dict[f] = xs_dict[f]
        return out_dict
        
    def processRescalePCA(self, all_data, reduced_dim = 3):
        '''Rescale the data, and then apply the PCA reduction'''
        self.saveFeatureOrder(all_data)
        self.getRescaleInfo(all_data)
        rescaled_data_dict = self.applyRescale(all_data)
        data_mat = self.dictToMat(rescaled_data_dict)
        self.PCAProcess = PCA(n_components= reduced_dim)
        self.PCAProcess.fit(data_mat)
        reduced_data_mat = self.PCAProcess.transform(data_mat)
        self.component_range = (np.min(reduced_data_mat,0), np.max(reduced_data_mat,0))
        self.roundSecond(self.component_range)
        
    def applyRescalePCA(self, other_data):
        '''Use the pre computed sacle and PCA info to process new data'''
        rescaled_data = self.applyRescale(other_data)
        rescaled_mat = self.dictToMat(rescaled_data)
        PCA_data = self.PCAProcess.transform(rescaled_mat)
        return PCA_data

    # helpers 
    def dictToMat(self, xs_dict):
        all_feas = [fea for fea in xs_dict]
        out_mat = np.zeros((len(xs_dict[all_feas[0]]), len(all_feas) - 1))
        for fea in all_feas:
            if fea != 'Time':
                out_mat[:, self.feature_order[fea]] = xs_dict[fea].T
        return out_mat

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