import numpy as np
import random
import pandas as pd
import time

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

#import past data into python 
class DataCombiner:
    def __init__(self):
        '''
        Assume all trials have the same fields, 
        each trial is stored as a pandas data frame
        trials are managed by the following format in a dict
        {TrialType_Trial#: pd.df ...}
        '''
        self.all_trials = {}
        self.all_trial_info = {}
        self.label_order = []

    def __str__(self):
        data_info = 'Fields and Trial Number' + str([(field, len(self.all_trials[field])) for field in self.all_trials])
        return data_info

    def loadTrialData(self, all_trial_info, data_folder_address, file_ext):
        ''' all_trial_info in the format {specific_trail:{'General_type', num_trials}, ...} '''
        ini_time = time.time()
        print('Start Loading: From txt files into Pandas')
        self.processTrialInfo(all_trial_info)
        # load the data into the format 
        for each_spec_trial in all_trial_info:
            cur_gen = all_trial_info[each_spec_trial][0]
            cur_max_trial_num = all_trial_info[each_spec_trial][1]
            for idx in range(1,cur_max_trial_num+1):
                trial_str = (('_0' if idx < 10 else '_') + str(idx))
                # add trial data under general - sepcific - list of trial 
                self.all_trials[cur_gen][each_spec_trial].append(ToPandasData(data_folder_address 
                                                                              + each_spec_trial + trial_str + file_ext))
        print('Finish Loading: Time taken ' + str(round(time.time() - ini_time,3)) + ' sec\n')
        
    def combineAllData(self):
        ini_time = time.time()
        print('Start Combining: Combine specific type trials under each general type')
        out = {}
        for each_gen in self.all_trial_info:
            out[each_gen] = {}
            specs = self.all_trials[each_gen]
            for each_spec in specs:
                cur_combine = self.combineSameSpecTrials(specs[each_spec])
                out[each_gen][each_spec] = cur_combine
        print('Finish Combining: Time taken ' + str(round(time.time() - ini_time,3)) + ' sec\n')
        return out
                
    # helpers 
    def processTrialInfo(self, all_trial_info):
        for each_spec in all_trial_info:
            cur_gen = all_trial_info[each_spec][0]
            if cur_gen not in self.all_trial_info: 
                self.all_trial_info[cur_gen] = set([each_spec])
                self.label_order.append(cur_gen)
                self.all_trials[cur_gen] = {}
            else: 
                self.all_trial_info[cur_gen].add(each_spec)
            if each_spec not in self.all_trials[cur_gen]: self.all_trials[cur_gen][each_spec] = []
            
    def combineSameSpecTrials(self, same_type_trials):
        '''
        change the time line for the trials, so that 0 -> tn, 0 -> tm becomes a continuous time line 
        combine list of trials together to a single pd.df with first idx being the trial type 
        '''
        last_end_time = 0
        for trial in same_type_trials:
            trial['Time'] += last_end_time
            last_end_time = np.mean(np.diff(trial['Time'])) + trial['Time'].iloc[-1]
        return pd.concat([pd.concat(same_type_trials, ignore_index=True)])


class DataAugmentor:
    def __init__(self, combined_trials):
        '''
        Data are orginized in the following format in a pd frame
        first idx: trial_type                row0 row1 .... Time
        second idx: actual idx for the data  v0,0  v2,0     t1
        '''
        self.original_data = combined_trials
        self.all_types = {}
        for each_gen in self.original_data:
            self.all_types[each_gen] = set()
            specs = self.original_data[each_gen]
            for each_spec in specs:
                self.all_types[each_gen].add(each_spec)
                
    def __str__(self):
        if not self.num_augmented: return str(self.all_types) + " " + self.num_augmented
        else: return 'No data augmentation is performed' + ' ' + str(self.all_types)
    
    def subSampleAll(self, out_length, num_draw, std_percentile = 10):
        ini_time = time.time()
        print('Start Data Augmentation: subsample data from combined data and add noise')
        out = {}
        for each_gen in self.all_types:
            cur_ini_time = time.time()
            out[each_gen] = []
            cur_num_draw = int(np.ceil(num_draw/len(self.all_types[each_gen])))
            specs_data = self.original_data[each_gen]
            for each_spec in specs_data:
                spec_data = specs_data[each_spec]
                sub_data = self.subSampleOneType(spec_data, out_length, cur_num_draw)
                out[each_gen].extend(self.addNoise(sub_data, std_percentile))
            print('Finish Augmenting Data: ' + each_gen + ', Time Taken ' + str(round(time.time() - cur_ini_time,3)) + ' sec')
        print('Finish All Data Augmentation, Time Taken: ' + str(round(time.time() - ini_time,3)) + ' sec\n')
        return out
    
    # helpers 
    def subSampleOneType(self, data_one_type, out_length, num_draw):
        out_subSamples = [None for _ in range(num_draw)]
        len_old_data = len(data_one_type['Time'])
        if out_length > out_length: raise ValueError('Original Data not enough to pull subsample with length' 
                                                     + str(out_length))
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
        if std_percentile == 0: return list_old_data
        out_noisy_data = [_ for _ in range(len(list_old_data))]
        for idx in range(len(list_old_data)):
            cur_old = list_old_data[idx]
            len_data = len(cur_old['Time'])
            new_data = cur_old.copy()
            mea_fields = [each for each in list(new_data) if each != 'Time']
            for each_field in mea_fields:
                cur_noise = np.random.normal(0, np.percentile(np.abs(cur_old[each_field]), std_percentile), len_data)
                new_data[each_field] += cur_noise
            out_noisy_data[idx] = new_data
        return out_noisy_data