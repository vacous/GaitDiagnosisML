import numpy as np
import random

# helper function to help parse the input file
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

#import past data into python 
def ToPythonData(old_address):
    reform_data = {}
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
    # convert all data into np array
    for h in headers: 
        out[h] = np.array(out[h])
        if h == 'Time': out[h] = out[h]/10**3.0
    return out


def CombineTrials(trials):  
    '''Assume all trials have the same fields, trials are stored in a dict'''
    combined = {}
    # scan through all trials and record the total data length 
    # total data length = len(np_1) + len(np_2) + ...
    len_counter = {}
    # save the current idx postion to add the next data after
    last_idx = {}
    for t in trials:
        trial = trials[t]
        for field in trial: 
            if field not in len_counter: len_counter[field] = len(trial[field])
            else: len_counter[field] += len(trial[field])
    for field in len_counter:
        combined[field] = np.zeros(len_counter[field])
        last_idx[field] = 0
    # add the each data into the large chunk 
    for t in trials:
        trial = trials[t]
        for field in trial:
            cur_last_idx = last_idx[field]
            cur_data_len = len(trial[field])
            if field != 'Time':
                combined[field][cur_last_idx: cur_last_idx + cur_data_len] = trial[field]
            else:
                cur_time = trial[field]
                avg_time_diff = np.mean(np.diff(cur_time))
                previous_end_time = combined[field][cur_last_idx-1 if cur_last_idx != 0 else 0]
                combined[field][cur_last_idx: cur_last_idx + cur_data_len] = cur_time + avg_time_diff + previous_end_time
            last_idx[field] += cur_data_len     
    return combined

def SubSampleData(old_long_data, required_length, num_draw):
    out_subsamples = [None for _ in range(num_draw)]
    len_old_data = len(old_long_data['Time'])
    for idx in range(num_draw):
        cur_rand_start = random.randint(0, len_old_data - required_length - 1)
        cur_subsample = {}
        for each_field in old_long_data:
            cur_field_subdata = old_long_data[each_field][cur_rand_start: cur_rand_start + required_length]
            if each_field != 'Time':
                cur_subsample[each_field] = cur_field_subdata
            else:
                cur_subsample[each_field] = cur_field_subdata - cur_field_subdata[0]
        out_subsamples[idx] = cur_subsample
    return out_subsamples


def AddNoise(old_data, std_percentile = 3):
    if std_percentile == 0: return old_data
    out_data = {}
    for each_f in old_data:
        cur_data = old_data[each_f]
        if each_f != 'Time':
            cur_noise = np.random.normal(0, np.percentile(np.abs(cur_data), std_percentile), len(old_data[each_f]))
            out_data[each_f] = np.array(cur_data) + cur_noise
        else:
            out_data[each_f] = np.array(cur_data)
    return out_data 