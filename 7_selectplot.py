from common import check_dir
from datetime import datetime
from mseed_record_info import mseed_record_info
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import sys
from scipy.stats import zscore
import scipy.signal as signal
import threading
import subprocess
import csv
import math
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

def save_array_to_csv(data_array, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data_array:
            for col in row:
                writer.writerow(col)

def bandpass_filter(data, lowcut, highcut, order=5):
    fs = 100
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = signal.butter(order, [low, high], btype='bandpass')
    y = signal.lfilter(b, a, data)
    return y

def moving_average(original_y, average_length):
    return np.convolve(original_y, np.ones(average_length, dtype=np.float32), "same") / average_length

def band_power(th, fft, band_start, band_end):
    Fs = 100
    N = np.shape(th)[0]
    n = np.shape(th)[1]
    PSD = np.abs(fft) ** 2
    freq = np.fft.fftfreq(N, 1/Fs)
    feature = np.sum(PSD[np.where((freq >= band_start) & (freq <= band_end))])
    # print(feature)
    return feature

def k_mean_cluster(band_power_db, num_cluster):
    max_iter = 1000
    kmeans_result = KMeans(num_cluster, init="k-means++", n_init=10, max_iter=max_iter).fit(band_power_db)
    labels = kmeans_result.labels_
    center = kmeans_result.cluster_centers_
    amount = []
    for i in range(num_cluster):
        amount.append([i, np.sum(labels==i)])
        print("Cluster ", i, " size: ", np.sum(labels==i))
    amount = np.array(amount)
    # sort size
    # amount = amount[amount[:,1].argsort()[::-1]]
    # print("Cluster size: ", amount)
    
    # print(labels)
    return kmeans_result, amount

def copyfile_subprocess(logging_path, dir_path):
    cmd = 'copy "%s" %s' %(logging_path, dir_path)
    print(cmd)
    status = subprocess.call(cmd, shell=True)
    if status != 0:
        if status < 0:
            print("terminated by signal", status)
        else:
            print("command failed, return code -", status)
    else:
        print("Execute %s \n" %cmd)

if __name__ == '__main__':
    len_argv = len(sys.argv)
    if len_argv >= 4:
        place_name = check_dir(sys.argv[1])
        place_name = place_name[:-1]
        source_dir  = check_dir(sys.argv[2])
        output_dir     = check_dir(sys.argv[3])
        time_length    = float(sys.argv[4])
        place_num = []
        if (place_name == "SYK"):
            # place_num = ["01", "02", "03", "04", "05", "06"]
            place_num = ["01"]
            # place_num = ["01", "02"]
        if (place_name == "YD"):
            place_num = ["04", "05", "06", "07", "21"]
            # place_num = ["04", "05", "06", "07"]
            # place_num = ["07"]

        # plt param
        plt.rcParams['agg.path.chunksize'] = 10000
        figsize_width                      = 2160
        figsize_height                     = 1440
        my_dpi                             = 180
        linewidth                          = 0.5
        band_start = 0.5
        band_end = 3
        num_cluster = 3

        
        time_num = int(((86400 / time_length) * 2 - 1) * 17)
        day_list = np.arange(3, 19+1)
        band_power_temp = np.zeros((time_num, 1))
        band_power_db = np.zeros((len(place_num), time_num))
        
        try:
            band_power_db = np.load(output_dir + str.upper(place_name) + "_band_power.npy")

        except:
            for p in range(len(place_num)):
                source_dir_fft = source_dir + str.lower(place_name) + place_num[p] + "_fft" + str(int(time_length)) + os.sep
                source_dir_th  = source_dir + str.lower(place_name) + place_num[p] + "_th" + str(int(time_length)) + os.sep

                print("source_dir_th  = " + source_dir_th)
                print("source_dir_fft = " + source_dir_fft)
                
                # lack = np.zeros((len(place_num), time_num))
                counter = 0
                for day in day_list:
                    
                    for t in range(int(((86400 / time_length) * 2 - 1))):
                        counter += 1
                        if (time_length == 3600) or (time_length == 1800):
                            np_filename = "2023.03." + str(day).zfill(2) + "_" + str(int((86400 / time_length) * 2 - 1)) + "-" + str(t).zfill(2) + "_" + str.lower(place_name) + place_num[p] + ".npy"
                        elif (time_length == 900):
                            np_filename = "2023.03." + str(day).zfill(2) + "_" + str(int((86400 / time_length) * 2 - 1)) + "-" + str(t).zfill(3) + "_" + str.lower(place_name) + place_num[p] + ".npy"
                        print("#" + str(counter) + "/" + str(time_num) + ": " + np_filename)
                        output_filename = output_dir + np_filename.replace("npy", "png")
                        
                        try:
                            # Plot FFT data
                            # fft      = np.load(source_dir_fft + np_filename)
                            th  = np.load(source_dir_th + np_filename)
                            
                            # th[:,1] = th[:,1] * np.hamming(len(th[:,1]))
                            fft = np.fft.fft(th, axis=0)
                            # band power
                            band_power_temp[counter-1] = band_power(th, fft, band_start, band_end)
                        except:
                            band_power_temp[counter-1] = np.nan
                            print("lack")

                band_power_temp_non_nan = zscore(band_power_temp[~np.isnan(band_power_temp)])
                band_power_temp[~np.isnan(band_power_temp)] = band_power_temp_non_nan
                band_power_db[p][:] = np.squeeze(band_power_temp)
                
            # save csv
            # np.savetxt(output_dir + str.upper(place_name) + "_band_power.csv", band_power_db, delimiter=",")
            # save np
            np.save(output_dir + str.upper(place_name) + "_band_power.npy", band_power_db)
        for j in range(int(band_power_db.shape[1])):
            band_power_db[np.isnan(band_power_db[:, j]), j] = np.nanmean(band_power_db[:, j])

        band_power_db_non_nan = np.array(band_power_db).T
        band_power_db_non_nan = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(band_power_db_non_nan)
        np.savetxt(output_dir + str.upper(place_name) + "_band_power.csv", band_power_db_non_nan.T, delimiter=",")
        print("band_power_db_non_nan.shape = ", band_power_db_non_nan.shape)
        # os.environ['OMP_NUM_THREADS'] = '4'
        kmeans_result, amount = k_mean_cluster(band_power_db_non_nan, num_cluster)
        # for am in range(num_cluster):
        #     if (amount[am][0] != am):
        #         am_error.append((amount[am][0], am))
        am_error = amount[amount[:,1].argsort()[::-1]]
        # amount[:,0] = np.arange(num_cluster)
        sorted_indices = amount[:, 1].argsort()[::-1]
        sorted_indices = sorted_indices.tolist()
        kmeans_labels_list = []
        for k in kmeans_result.labels_:
            kmeans_labels_list.append(sorted_indices.index(k))
        kmeans_labels = np.array(kmeans_labels_list)

        kmeans_centers = kmeans_result.cluster_centers_

        # read and write csv
        band_power_read = np.loadtxt(output_dir + str.upper(place_name) + "_band_power.csv", delimiter=",")
        band_power_labeled = np.vstack((np.array(band_power_read), kmeans_labels.T))

        label_idx = {}
        for i in range(0, num_cluster):
            label_idx[i] = np.where(kmeans_labels.T == i)
            # for j in range(0, len(label_idx[i])):
            #     label_idx[i] = str(label_idx[i][j]/(int((86400 / time_length) * 2 - 1))).zfill(2)+str(label_idx[i][j]%(int((86400 / time_length) * 2 - 1)))
        
        print("label_idx = ", label_idx)
        for p in range(len(place_num)):
            for i in range(0, num_cluster):
                for j in label_idx[i]:
                    for k in j:
                        try:
                            source_dir_th  = source_dir + str.lower(place_name) + place_num[p] + "_th" + str(int(time_length)) + os.sep
                            np_filename = "2023.03." + str(((k/(int((86400 / time_length) * 2 - 1)))+3).astype(np.int64)).zfill(2) + "_" + str(int((86400 / time_length) * 2 - 1)) + "-" + str((k%((86400 / time_length) * 2 - 1)).astype(np.int64)).zfill(2) + "_" + str.lower(place_name) + place_num[p] + ".npy"
                            # print(np_filename)
                            logging_path = "D:\\Infrasound\\plot_new" + os.sep + np_filename.replace("npy", "png")
                            dir_path = "D:\\Infrasound\\db\\band_power\\selected" + os.sep + str(i) + os.sep + np_filename.replace("npy", "png")
                            # print(dir_path)
                            copyfile_subprocess(logging_path, dir_path)
                            
                        except FileNotFoundError:
                            print("Broken data")
                            continue                 

    else:
        # print("    python3 " + sys.argv[0] + " <source_dir_th> <source_dir_fft> <output_dir> <time_length (seconds)>")
        # print("Ex. python3 " + sys.argv[0] + " /home/aries/Working/Infrasound/db/th_3600/ /home/aries/Working/Infrasound/db/fft_3600/ /home/aries/Working/Infrasound/images/th_fft_3600/ 3600")
        print("Ex. python " + sys.argv[0] + " SYK /home/aries/Working/Infrasound/db/th+fft /home/aries/Working/Infrasound/images/band_power/ 3600")
        os.sys.exit(0)


        # python 7_selectplot.py SYK D:\Infrasound\db\th+fft D:\Infrasound\db\band_power 3600
