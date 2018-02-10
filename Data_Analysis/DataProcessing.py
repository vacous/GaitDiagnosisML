import numpy as np
import random
import pandas as pd

# helper function to help parse the input file
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

#import past data into python 
def ToPandasData(old_address):
    pandas_data = pd.DataFrame()
    with open(old_address, 'rb') as file:
        out = {}
        headers = [txt.decode('UTF-8', 'ignore') for txt in file.readline().split()]
        num_points = int(file.readline().split()[0])
        for h in headers: out[h] = []
        for line in file:
            # remove corrupted data 
            if all(isfloat(v) for v in line.split()) and len(line.split()) == len(headers):
                vals = [float(v) for v in line.split()]
                for idx in range(len(vals)):
                    out[headers[idx]].append(vals[idx])
    for h in headers: 
        out[h] = np.array(out[h])
        # convert Time data into seconds 
        if h == 'Time': 
        	out[h] = out[h]/10**3.0
        elif h == 'Voltage_01' or h == 'Voltage_02':
        	cur_min, cur_max = np.min(out[h]), np.max(out[h])
        	out[h] = (out[h] - cur_min)/(cur_max - cur_min)
    # collect all data into a pandas data frame
        pandas_data[h] = out[h]
    return pandas_data   

class DataCombiner:
    def __init__(self):
        '''
        Assume all trials have the same fields, 
        each trial is stored as a pandas data frame
        trials are managed by the following format in a dict
        {TrialType_Trial#: pd.df ...}
        '''
        self.all_trials = {}
    def __str__(self):
        data_info = 'Fields and Trial Number' + str([(field, len(self.all_trials[field])) for field in self.all_trials])
        return data_info
    
    def addNewTrial(self, trial, trial_type):
        if trial_type not in self.all_trials: self.all_trials[trial_type] = [trial]
        else: self.all_trials[trial_type].append(trial)
            
    def combineAllData(self):
        result_buffer = {}
        for each_type in self.all_trials:
            result_buffer[each_type] = self.combineSameTypeTrials(self.all_trials[each_type], each_type)
        return self.combineDiffTypeTrials(result_buffer)
            
    # helpers 
    def combineSameTypeTrials(self, same_type_trials, trial_type):
        '''
        change the time line for the trials, so that 0 -> tn, 0 -> tm becomes a continuous time line 
        combine list of trials together to a single pd.df with first idx being the trial type 
        '''
        last_end_time = 0
        for trial in same_type_trials:
            trial['Time'] += last_end_time
            last_end_time = np.mean(np.diff(trial['Time'])) + trial['Time'].iloc[-1]
        return pd.concat([pd.concat(same_type_trials, ignore_index=True)], keys=[trial_type])
        
    def combineDiffTypeTrials(self, diff_type_pddfs_dict):
        return pd.concat([diff_type_pddfs_dict[each_type] for each_type in diff_type_pddfs_dict])


class DataAugmentor:
    def __init__(self, combined_types_df):
        '''
        Data are orginized in the following format in a pd frame
        first idx: trial_type                row0 row1 .... Time
        second idx: actual idx for the data  v0,0  v2,0     t1
        '''
        self.original_data = combined_types_df
        self.all_types = set(combined_types_df.index.levels[0])
        self.num_augmented = None
        
    def __str__(self):
        if not self.num_augmented: return str(self.all_types) + " " + self.num_augmented
        else: return 'No data augmentation is performed' + ' ' + str(self.all_types)
    
    def subSampleAll(self, out_length, num_draw, std_percentile = 3):
        all_sub_samples = {}
        for each_type in self.all_types:
            sub_samples = self.subSampleOneType(self.original_data.loc[each_type], out_length, num_draw)
            sub_sampled_noise = self.addNoise(sub_samples, std_percentile)
            all_sub_samples[each_type] = sub_sampled_noise
        return all_sub_samples
    
    # helpers 
    def subSampleOneType(self, data_one_type, out_length, num_draw):
        out_subSamples = [None for _ in range(num_draw)]
        len_old_data = len(data_one_type['Time'])
        if out_length > out_length: raise ValueError('Original Data not enough to pull subsample with length' + str(out_length))
        for idx in range(num_draw):
            cur_rand_start = random.randint(0, len_old_data - out_length - 1)
            cur_range = range(cur_rand_start,cur_rand_start + out_length)
            cur_sub = data_one_type.iloc[cur_range].copy()
            cur_sub = cur_sub.reset_index(drop = True)
            cur_sub['Time'] = cur_sub['Time'] - cur_sub['Time'][0]
            out_subSamples[idx] = cur_sub
        return out_subSamples
            
    def addNoise(self, list_old_data, std_percentile):
        '''
        add noise to the sub sampled data
        '''
        if std_percentile == 0: return old_data
        out_noisy_data = []
        for idx in range(len(list_old_data)):
            cur_old = list_old_data[idx]
            len_data = len(cur_old['Time'])
            new_data = cur_old
            for each_field in [each for each in list(new_data) if each != 'Time']:
                cur_noise = np.random.normal(0, np.percentile(np.abs(cur_old[each_field]), std_percentile), len_data)
                new_data[each_field] += cur_noise
            out_noisy_data.append(new_data)
        return out_noisy_data