from common import check_dir
from datetime import datetime
from mseed_record_info import mseed_record_info
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.stats import zscore
import scipy.signal as signal
import threading
import csv


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
    freqs, psd = signal.welch(th, Fs, nperseg=N)
    return freqs, psd


if __name__ == '__main__':
    len_argv = len(sys.argv)
    if len_argv >= 4:
        place_name = check_dir(sys.argv[1])
        place_name = place_name[:-1]
        source_dir  = check_dir(sys.argv[2])
        output_dir     = check_dir(sys.argv[3])
        time_length    = float(sys.argv[4])
        place_num = []

        # plt param
        plt.rcParams['agg.path.chunksize'] = 10000
        figsize_width                      = 2160
        figsize_height                     = 1440
        my_dpi                             = 180
        linewidth                          = 0.5

        if (place_name == "SYK"):
            place_num = ["01", "02", "03", "04", "05", "06"]
            # place_num = ["04"]
            # place_num = ["01", "02", "03", "04"]
        if (place_name == "YD"):
            # place_num = ["04", "05", "06", "07", "21"]
            place_num = ["21"]
        time_num = int(((86400 / time_length) * 2 - 1) * 17)
        day_list = np.arange(3, 19+1)
        band_power_temp = np.zeros((time_num, 1))
        band_power_db = np.zeros((len(place_num), time_num))

        try:
            band_power_db = np.load(output_dir + str.upper(place_name) + "_band_power.npy")

        except:
            for p in range(len(place_num)):
                source_dir_fft = source_dir + str.lower(place_name) + place_num[p] + "_fft" + str(int(time_length)) + "\\"
                source_dir_th  = source_dir + str.lower(place_name) + place_num[p] + "_th" + str(int(time_length)) + "\\"

                print("source_dir_th  = " + source_dir_th)
                print("source_dir_fft = " + source_dir_fft)
                # place_name = str.upper(source_dir_th[source_dir_th.rfind("th+fft\\")+7:source_dir_th.rfind("_")])
                th_file_list   = os.listdir(source_dir_th)
                th_file_list.sort()
                fft_file_list  = os.listdir(source_dir_fft)
                fft_file_list.sort()
                # band = [[0.1, 1], [1, 10], [10, 20]]
                band = [[0.001, 0.01], [0.01, 0.1]]
                print(time_num)
                print(len(band))
                
                # lack = np.zeros((len(place_num), time_num))
                counter = 0
                psd_sum = []
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
                            fft      = np.load(source_dir_fft + np_filename)
                            th  = np.load(source_dir_th + np_filename)
                            # fft = np.fft.fft(th[:,1], axis=0)
                            # hamming window
                            # fft[:,1] = fft[:,1] * np.hamming(len(fft[:,1]))
                            # band power
                            # freq, psd = band_power(th[:,1], fft, band_start=0.1, band_end=1)
                            
                            if (len(th[:,1])<12000):
                                th = np.pad(th, [(0,12000-len(th[:,1])),(0,0)], 'constant', constant_values=0)
                            f, psd = signal.welch(th[:,1], fs=100, nperseg=12000, window='hamming', scaling='density')
                            
                            # f_mean, psd_mean = signal.welch(th[:,1], fs=100, nperseg=12000, window='hamming', scaling='density', average='median')
                            # plt.semilogy(f, psd)
                            # plt.semilogy(f_mean, psd_mean)
                            # plt.xlabel('frequency [Hz]')
                            # plt.ylabel('dB')
                            # plt.xlim([10, 15])
                            
                            psd_sum.append(psd)
                            print(np.array(psd_sum).shape)
                            # plt.show()
                            # plt.close()
                        except:
                            band_power_temp[counter-1] = np.nan
                            print("lack")
                psd_avg = np.nanmean(np.array(psd_sum), axis=0)
                plt.figure(figsize=(figsize_width/my_dpi, figsize_height/my_dpi), dpi=my_dpi)
                for s in range(np.array(psd_sum).shape[0]):
                    plt.semilogy(f, np.array(psd_sum)[s,:], alpha=0.5, color='gray', linewidth=linewidth*0.25)
                plt.semilogy(f, psd_avg)
                plt.xlabel('frequency [Hz]')
                plt.ylabel('dB')
                plt.savefig(output_dir + str.upper(place_name) + place_num[p] + "_PSD.png", dpi=my_dpi)
                plt.close()
    else:
        # print("    python3 " + sys.argv[0] + " <source_dir_th> <source_dir_fft> <output_dir> <time_length (seconds)>")
        # print("Ex. python3 " + sys.argv[0] + " /home/aries/Working/Infrasound/db/th_3600/ /home/aries/Working/Infrasound/db/fft_3600/ /home/aries/Working/Infrasound/images/th_fft_3600/ 3600")
        print("Ex. python " + sys.argv[0] + " SYK /home/aries/Working/Infrasound/db/th+fft /home/aries/Working/Infrasound/images/band_power/ 3600")
        os.sys.exit(0)


        # python 6c_psd.py YD D:\Infrasound\db\th+fft D:\Infrasound\db\band_power 900