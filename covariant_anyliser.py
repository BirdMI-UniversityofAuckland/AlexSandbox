
# coding: utf-8

# In[ ]:


"""
Covariance Analyser

Takes and anaesthetised Zibra Finch's HVC neuronal potential spike-time data 
and it's coresponding birds own song audio and returns a data-structure
containing covarient elements.

This program will hopfully be transferable to and awake birds HVC spike times
and the corresponding sound output, thereby can be used to inform a BMI
which neural activity related to which changes in song output

Functions: ...

Classes: ...

Exceptions: ...

Created on Mon Jun 29 13:57:37 2015

@author: Alex Brebner
"""


# In[313]:

##Imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sp
import scipy.io.wavfile as wav
import scipy.signal as sgnp
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import fftpack
from scipy import stats
import os
import csv

get_ipython().magic(u'matplotlib inline')


# In[2]:

#Constants
sampling_freq = 44100
adult_birds = set(['blk161', 'o222', 'r87', 'w293',]) #names of the adult birds used in Crandall 2007
good_electrodes = {'blk161': ['e2'], 'o222': ['e1', 'e2'], 'r87': ['e1', 'e2'], 'w293': ['e2']}

#work
data_dir = 'C:\\Users\\abre049.UOA\\Documents\\data\\Crandall 2007' 
r87 = 'C:\\Users\\abre049.UOA\\Documents\\data\\Crandall 2007\\r87\\sngr870119wtrig02-Apr135757\\sng162004_0119v4'
blk161 = 'C:\\Users\\abre049.UOA\\Documents\\data\\Crandall 2007\\blk161\\sngblk1610308wtrigC23-Mar140834\\sng163839_0308v3.mat'
blk161_syl_path = 'C:\\Users\\abre049.UOA\\Documents\\data\\blk161_syll_clust.csv'
blk161_feat_dir = 'C:\\Users\\abre049.UOA\\Documents\\data\\blk161_feature_data'

#home
# data_dir = 'C:\\Users\\Alex\\Documents\\Crandall2007data'
# blk161 = 'C:\\Users\\Alex\\Documents\\Crandall2007data\\blk161\\sngblk1610308wtrigC23-Mar140834\\sng163839_0308v3.mat'


# #Load the data location
# 
# I want to have a way of feeding the data into python in such a way that I can easy change the scope of data. ie change from single 2-15sec recording to all the data from all birds. 
# 
# The data is stored like this: 
# Crandall 2007[bird_name_x[...], bird_name_y[day_a[...], day_b[single_song_data_i, single_song_data_j, ...], ...], ...]
# 
# To do this I will make a function which will iterate through all the folders and files and store each subdirectory as dictionary. This can then be used to generate the urls of all data files

# In[120]:

def data_file_urls(data_dir):
    """
    Takes the path name of the directory containing bird song data returning a
    a dictionary of all songs.
    Parameters:
        data_dir = path name of the directory containing data
    Return object:
        Dictionary where {dir: 'data_dir', bird: {day: set(['song_url'])}}.
    """
    data_urls = {} # dict that will be returned
    
    for bird in os.listdir(data_dir): # finds bird folders
        if bird != '__MACOSX': # ignores metadata for macs
            data_urls[bird] = {}
    for bird in data_urls.iterkeys():
        for day in os.listdir(os.path.join(data_dir, bird)): # finds day folders
            if day != '.DS_Store':
                data_urls[bird][day] = set(os.listdir(os.path.join(data_dir, bird, day))) # finds files and compiles dictionaries
                data_urls[bird][day].discard('wavs')
    data_urls['dir'] = data_dir
    return data_urls
    
song_locs = data_file_urls(data_dir)
print song_locs['blk161'].keys()


# In[5]:

#Classes


# In[20]:

#Master function


# In[22]:

#Helper functions


# #Path generator

# In[127]:

def gen_path(bird, song):
    if song[-4:] == '.wav':
        file_name = 'sng' + song[6:12] + '_' + song[1:5] + song[12:-4] + '.mat'
    elif song[-4:] == '.mat':
        file_name = song
    else:
        return 'Song name not .wav or .mat'
    for folder in song_locs[bird].keys():
        if folder[9:13] == file_name[10:14]:
#             print folder[9:13], file_name[10:14]
            if file_name in song_locs[bird][folder]:
                folder_name = folder
                break
    return data_dir + '\\' + bird + '\\' + folder_name + '\\' + file_name
        
# song = 'v0308t091922v3.wav'
# print 'sng' + song[6:12] + '_' + song[1:5] + song[12:14] + '.mat'
#sng082936_0308v3.mat
print gen_path('blk161', 'v0308t080339v10.wav')
# print gen_path('blk161', 'sng091922_0308v3.mat')
# sngblk1610308wtrigB23-Mar134644


# In[23]:

#def sound_variations(sound_data):
    '''
    takes sound_data in X format and returns a ?dictionary where 
    keys = str(name_of_variable) and values = float(coefficient_of_variability)
    '''
#   need to know how sounds can varry


# In[24]:

#def spike_variations(spike_data):
    '''
    takes spike_data in the format X and returns a ?dictionary where
    Keys = str(name_of_variable) and values = float(coefficient_of_variability)
    '''
#    Need to know how spike times can varry


# #What does the raw data look like?

# In[7]:

#Workspace
##data_url is now out of date, needs fix
def display_raw_data(data):
    print 'Data class: ', type(data)
    print 'Keys: ', data.keys()
    print 'Header: ', data['__header__']
    print 'Globals: ', data['__globals__']
    print 'Versions: ', data['__version__']
    print len(data['sng']) #74901

    plt.plot(data['sng'], label='sng (audio)', linewidth=0.5)
    plt.plot(data['ref'][0], label='ref', linewidth=0.5)
    plt.plot(data['e1'][0], label='e1', linewidth=0.5)
    plt.plot(data['e2'][0], label='e2', linewidth=0.5)
    plt.plot(data['e3'][0], label='e3', linewidth=0.5)
    plt.legend()
    plt.title('Raw Data plot')
    plt.ylabel('Signal / mV')
    plt.xlabel('Data point')

display_raw_data(sp.loadmat(blk161))


# #Listen to audio

# In[26]:

def make_wav_file(data):
    amplify_data = np.array([[int((idx[0] * 1800000))] for idx in data])
    print amplify_data
    sp.wavfile.write('audio2.wav', sampling_freq, data)
    
# print sp.loadmat(blk161)['sng']
# make_wav_file(sp.loadmat(blk161))


# #What does one population electrode look like?

# In[307]:

def plot_electrode(data, electrode, lbl=False, rng=[0,-1], plot=True):
    if lbl == False:
        lbl = electrode
    e_array = data[electrode][0][rng[0]:rng[1]]
    if plot == True:
        plt.plot(e_array, label=lbl, linewidth=0.5)
        plt.legend()
        plt.title('Electrode plot')
        plt.ylabel('Signal / mV')
        plt.xlabel('Data point')
    return e_array

plot_electrode(sp.loadmat(blk161), 'e2')
plot_electrode(sp.loadmat(blk161), 'ref')
# plot_electrode(sp.loadmat(blk161), 'e1')

# plot_electrode(sp.loadmat(blk161), 'e3')
plt.show()

e2 = np.array(plot_electrode(sp.loadmat(blk161), 'e2', plot=False))
ref = np.array(plot_electrode(sp.loadmat(blk161), 'ref', plot=False))
e2_sub_ref = e2 - ref
e2_div_ref = e2 / ref
plt.plot(e2_sub_ref, label='sub', lw=0.5)
plt.plot(e2_div_ref, label='div', lw=0.5)
plt.legend()
plt.show()


# #Plot song

# In[9]:

def plot_song(data, lw=0.5, plot_now=False):
    plt.plot(data['sng'], label='song', linewidth=lw)
    if plot_now:
        plt.legend()
        plt.title('Song plot')
        plt.ylabel('Signal / mV')
        plt.xlabel('Data point')
        plt.show()
    
plot_song(sp.loadmat(blk161), plot_now=True)


# #Plot spectrogram

# In[10]:

def plot_spectrogram(data):
    one_d_data = [idx[0] for idx in data]
    plt.specgram(one_d_data, Fs=sampling_freq)
    
plot_spectrogram(sp.loadmat(blk161)['sng'])
plt.show()


# #Import syllable data

# In[144]:

def get_syl_data(syll_data_path):
    '''
    Takes the directories of the .cvs file with syllable data and cluster number (produced with SAP2011 backup function).
    Return a 2d list with data
    Parameters:
        syll_data_dir - a directory containing the .cvs file produced by SAP2011
            with cols of song_name, start time, duration, mean_freq etc
    Returns:
        list of lists. first row contains the headings of the columns as strings. The remaining lists are the data as either
            strings or floats
    Exceptions: nil
    '''
    syl_data = [] # the return object
    with open(syll_data_path, 'r') as f: # opens the .csv file
        reader = csv.reader(f)
        headings = [] # will become the first row
        row_data = [] # used to make the table body
        row_numb = 0 # counter to keep track of which row as some data must be excluded as it is garbage
        for row in reader: # iterates through the .csv rows
            if row_numb >= 16 and row_numb < 61: # these row contain the heading information
                char_state = 0 # there is garbage characters around the useful information, this keeps track of the useful info
                col_head = ''
                for char in row[0]:
                    if char == "`": # data between the first two ` is the useful information
                        char_state += 1
                    elif char_state == 1: # only use useful info
                        col_head += char
                headings.append(col_head)
            elif row_numb == 61: # there are a few blank rows, use the first one to assign the headings into syl_data
                syl_data.append(headings)
            elif row_numb >= 70 and len(row) > 1: # identify the data body, ignoring empty rows
                for val in row: # cleans the data as there are useless chars arond the useful info
                    if val[0:3] == "  (":
                        row_data.append(float(val[3:])) #convert the numbers from strings to floats
                    elif val[:2] == " '":
                        row_data.append(val[2:-1]) # some info are strings
                    elif len(val) > 0:
                        row_data.append(float(val[1:])) # convert the numbers from strings to floats
            if len(row_data) > 0: # don't add empty rows to data
                syl_data.append(row_data)
            row_numb += 1
            row_data = []
        f.close()
    return syl_data

syl_data = get_syl_data(blk161_syl_path)
print syl_data[0:3]
# for syl in syl_data:
#     print syl[0]
    


# #Import song feature arrays

# In[12]:

def get_song_feat(path):
    '''
    Takes the path of .csv file produced by SAP2011 with the backup function which contains the set of arrays of song features
    Returns a dicitonary where keys are the feature name and values are a list of plots.
    Parameters:
        path of feature_table.csv
    Return:
        dict where {'song feature': [values]}
    Exceptions:
        nil
    '''
    feat_arrays = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        row_numb = 0
        dict_body = []
        for row in reader:
            if row_numb == 0:
                headings = row
                for en in headings:
                    dict_body.append([])
            elif row_numb > 1: # skip row two
                for idx in range(len(row)):
                    dict_body[idx].append(float(row[idx]))
            row_numb += 1
        f.close()
    for idx in range(len(headings)):
        feat_arrays[headings[idx]] = dict_body[idx]
    return feat_arrays
    
# testing = get_song_feat(blk161_feat_dir + '\\v0307t072941v2.csv')
# for y in testing:
#     if y != 'Time':
#         plt.plot(testing['Time'], testing[y], label=y)
# plt.legend()
# plt.show()
# plt.plot(testing['Amplitude'])
# plt.show()

testing = get_song_feat(blk161_feat_dir + '\\v0307t072941v2.csv')
print testing.keys()


# #Cut feature arrays to match syllable time

# In[17]:

def cut_array(array_times, array_vals, start_time, duration):
    start_idx = 0
    stop_idx = 0
    for idx in range(len(array_times)):
        if array_times[idx] <= start_time - 0.1: # note that because the values to not match exactly the selection is rounded down to the hundreth of a microsecond
            start_idx = idx
        elif array_times[idx] <= start_time + duration - 1:# note that because the values to not match exactly the selection is rounded up
            stop_idx = idx
    return array_times[start_idx: stop_idx], array_vals[start_idx: stop_idx]
            
syl_data = get_syl_data(blk161_syl_path)
feat_data = get_song_feat(blk161_feat_dir + "\\" + syl_data[1][-2][:-3] + 'csv')
x_vals, y_vals = cut_array(feat_data['Time'], feat_data['Amplitude'], syl_data[1][3], syl_data[1][4])
plt.plot(x_vals, y_vals)
feat_data = get_song_feat(blk161_feat_dir + "\\" + syl_data[5][-2][0:-3] + 'csv')
x_vals, y_vals = cut_array(feat_data['Time'], feat_data['Amplitude'], syl_data[5][3], syl_data[5][4])
plt.plot(x_vals, y_vals)
plt.show()


# #Center x values

# In[18]:

def center_x(x_vals, units='ms'):
    new_x = []
    left_shift = float(min(x_vals)) + ((max(x_vals) - min(x_vals)) * 0.5)
    if units='samples':
        for val in x_vals:
            new_x.append((val - left_shift) / sampling_freq * 0.001)
    elif units == 'ms':
        for val in x_vals:
            new_x.append(val - left_shift)
    else:
        return 'invalid argument for units, need "ms" or "samples"'
    return new_x
        
print center_x([1,2,3,4,5])
    


# #Plot a feature of a cluster

# In[146]:

def clust_gen(clust_num, feature, plot=False):
    '''
    generator returning a 2d list of centered x and y vals of the feature for the cluster specified. Can also plot arrays
    '''
    syl_data = get_syl_data(blk161_syl_path)
    sylls = []
    for idx in range(len(syl_data[0])):
        if syl_data[0][idx] == 'cluster':
            cluster_idx = idx
    for idx in range(len(syl_data[1:]) + 1):
        if syl_data[idx][cluster_idx] == clust_num:
            sylls.append(idx)
    for idx in sylls:
        feat_data = get_song_feat(blk161_feat_dir + "\\" + syl_data[idx][-2][0:-3] + 'csv') # look up data of the syllable
        x_vals, y_vals = cut_array(feat_data['Time'], feat_data[feature], syl_data[idx][3], syl_data[idx][4]) # extract array of that syllable from feature data
        x_vals = center_x(x_vals)
        yield [x_vals, y_vals, idx]
        if plot:
            plt.plot(x_vals, y_vals)
    if plot:
        plt.show()
        
    #clust <-- iterate through syll data and identify all syl of 1 clust
#     for syl in 
    #iterate though cluster and cut all the feature arrays of each syl
    #center all feature arrays
    #plot all arrays

x = []
y = []
for pairs in clust_gen(3, 'Amplitude', plot=True):
    x.append(pairs[0])
    y.append(pairs[1])
print len(x), len(y)


# #Match X values to median

# In[207]:

def match_x(ref_x, x_array, y_array):
    temp = ius(x_array, y_array)
    new_y = temp(ref_x)
    return new_y
    
x = []
y = []
for pairs in clust_gen(3, 'Amplitude'):
    x.append(pairs[0])
    y.append(pairs[1])
median_y = make_median_array(x, y, plot=True, average='median')
a, b =  match_x(median_x, x[0], y[0])
print 'old x len', len(x[0]), ', first val', x[0][0], ', last val', x[0][-1]
print 'new x len', len(a), ', first val', a[0], ', last val', a[-1]
print 'median x len', len(median_x), ', first val', median_x[0], ', last val', median_x[-1]
plt.plot(x[0][100:150], y[0][100:150], label='old')
plt.plot(a[100:150], b[100:150], label='new')
plt.legend()
plt.show()


# #Make median array

# In[161]:

def shortest_array(x_arrays):
    shortest_idx = -1
    shortest_val = 999999999999.9
    for idx in range(len(x_arrays)):
        if x_arrays[idx][-1] < shortest_val:
            shortest_idx = idx
            shortest_val = x_arrays[idx][-1]
    return x_arrays[shortest_idx]


# In[269]:

def make_median_array(x_array, y_arrays, plot=False, average='median'):
    '''
    takes list of lists with respective x and y values of a graph and finds the median (or mode) of y value at each x value. 
    Arrays must be of the same length and value (ie interpolated)
    Input
    '''
    median_y = []
    for idx in range(len(x_array)):
        y_vals = []
        for array in y_arrays:
            y_vals.append(array[idx])
        if average == 'median':
            median_y.append(np.median(y_vals))
        elif average == 'mean':
            median_y.append(np.mean(y_vals))
    if plot == True:
        plt.plot(x_array, median_y)
        plt.show()
    return median_y
    
x = []
y = []
for pairs in clust_gen(3, 'Amplitude'):
    x.append(pairs[0])
    y.append(pairs[1])
shortest_x_array = shortest_array(x)
for idx in range(len(x)):
    y[idx] = match_x(shortest_x_array, x[idx], y[idx])[1]
plt.plot(shortest_x_array, y[0])
plt.plot(shortest_x_array, y[1])
plt.show()
print len(y[0]), len(y[1])
foo = make_median_array(shortest_x_array, y, plot=True, average='mean')
bar = make_median_array(shortest_x_array, y, plot=True, average='median')


# #Quantify varience

# In[202]:

def calc_variance(ref, array):
    return np.corrcoef([ref, array])[0][1]

x = []
y = []
for pairs in clust_gen(3, 'Amplitude'):
    x.append(pairs[0])
    y.append(pairs[1])
shortest_x_array = shortest_array(x)
for idx in range(len(x)):
    y[idx] = match_x(shortest_x_array, x[idx], y[idx])[1]
median_y = make_median_array(shortest_x_array, y, average='median')
print calc_variance(median_y, y[0])
print calc_variance(median_y, y[1])
print calc_variance(median_y, y[2])
print ":)"


# #Do the same for electrodes

# #Extract electrode data
# need to add code to deal with birds with 2 good electrodes
# need to add code to normalise data to electrodes

# In[309]:

def get_elec_data(song_data, syl_start, syl_dur, preceed=45, electrode='e2'):
    e_data = song_data[electrode][0]
    ref_data = song_data['ref'][0]
    pre_sample = int((syl_start - preceed) * sampling_freq * 0.001)
    first_sample = int((syl_start) * sampling_freq * 0.001)
    last_sample = int((syl_start + syl_dur) * sampling_freq * 0.001)
    x_vals = center_x(range(first_sample, last_sample + 1))
    x_vals = range(int(x_vals[0] - int(preceed * sampling_freq * 0.001)), int(x_vals[0])) + x_vals
    x_in_ms = []
    for val in x_vals:
        x_in_ms.append(val / float(sampling_freq) * 1000.0)
    e_array = np.array(e_data[pre_sample: pre_sample + len(x_in_ms)])
    r_array = np.array(ref_data[pre_sample: pre_sample + len(x_in_ms)])
    return x_in_ms, (e_array - r_array)
    
print syl_data[1][-2], syl_data[1][3], syl_data[1][4]
x_vals1, e_array1 = get_elec_data(sp.loadmat(gen_path('blk161', syl_data[1][-2])), syl_data[1][3], syl_data[1][4], electrode='e2')
print len(x_vals1), len(e_array1)
# plt.subplot(211)
plt.plot(x_vals1, e_array1)
# plt.subplot(212)
x_vals2, e_array2 = get_elec_data(sp.loadmat(gen_path('blk161', syl_data[5][-2])), syl_data[5][3], syl_data[5][4], electrode='e2')
plt.plot(x_vals2, e_array2)
plt.show()

x = []
y = []
for a, b, idx in clust_gen(3, 'Amplitude'):
    syll = syl_data[idx]
    x_vals, y_vals = get_elec_data(sp.loadmat(gen_path('blk161', syll[-2])), syll[3], syll[4])
    x.append(x_vals)
    y.append(y_vals)
shortest_x_array = shortest_array(x)
for idx in range(len(x)):
    y[idx] = match_x(shortest_x_array, x[idx], y[idx])
median_y = make_median_array(shortest_x_array, y, average='median')
for array in y:
    plt.plot(shortest_x_array, array)
plt.plot(shortest_x_array, median_y)
plt.show()
print calc_variance(median_y, y[0])


# #Scattergram of electrode variance vs feature variance for each motif

# In[310]:

def calc_covariance(bird, clust, feature, plot=False):
    coefficients = [[],[]]
    # feature arrays
    feat_x = []
    feat_y = []
    e_x = []
    e_y = []
    for syl in clust_gen(clust, 'Amplitude'):
        feat_x.append(syl[0])
        feat_y.append(syl[1])
        temp_syl_data = syl_data[syl[2]]
        temp_e_x, temp_e_y = get_elec_data(sp.loadmat(gen_path(bird, temp_syl_data[-2])), 
                                           temp_syl_data[4], temp_syl_data[4], electrode=good_electrodes[bird][0])
        e_x.append(temp_e_x)
        e_y.append(temp_e_y)
    shortest_feat_x = shortest_array(feat_x)
    shortest_e_x = shortest_array(e_x)
    for idx in range(len(e_x)):
        feat_y[idx] = match_x(shortest_feat_x, feat_x[idx], feat_y[idx])
        e_y[idx] = match_x(shortest_e_x, e_x[idx], e_y[idx])
    feat_ave_y = make_median_array(shortest_feat_x, feat_y, average='median')
    e_ave_y = make_median_array(shortest_e_x, e_y, average='median')
    
    for idx in range(len(feat_x)):
        coefficients[0].append(calc_variance(feat_ave_y, feat_y[idx]))
        coefficients[1].append(calc_variance(e_ave_y, e_y[idx]))
    
    return coefficients

import time

covariance_coef = calc_covariance('blk161', 3, 'Amplitude')
print 'done'
# print covariance_coef


# In[318]:

plt.scatter(covariance_coef[0], covariance_coef[1])
plt.show()

slope, intercept, r_value, p_value, std_err = stats.linregress(covariance_coef[0], covariance_coef[1])
print slope, intercept, r_value, p_value, std_err

