from common import check_dir
from datetime import datetime
from mseed_record_info import mseed_record_info
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.stats import zscore
import threading



def plt_savefig(ouptut_image_full_name, my_dpi):
    output_folder = ouptut_image_full_name[:ouptut_image_full_name.rfind(os.sep)+1]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(ouptut_image_full_name, dpi=my_dpi)



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

    

if __name__ == '__main__':
    len_argv = len(sys.argv)
    if len_argv >= 4:
        source_dir_th  = check_dir(sys.argv[1])
        source_dir_fft = check_dir(sys.argv[2])
        output_dir     = check_dir(sys.argv[3])
        time_length    = float(sys.argv[4])
        th_file_list   = os.listdir(source_dir_th)
        th_file_list.sort()
        fft_file_list  = os.listdir(source_dir_fft)
        fft_file_list.sort()
        plt.rcParams['agg.path.chunksize'] = 10000
        figsize_width                      = 2160
        figsize_height                     = 1440
        my_dpi                             = 180
        linewidth                          = 0.5
        color                              = "blue"
        counter                            = 0
        band_power_db_1 = np.zeros((len(th_file_list), 1))
        band_power_db_2 = np.zeros((len(th_file_list), 1))
        band_power_db_3 = np.zeros((len(th_file_list), 1))

        for th_file in th_file_list:
            counter += 1
            print("#" + str(counter) + "/" + str(len(th_file_list)) + ": " + th_file)
            output_filename = output_dir + th_file.replace("npy", "png")
            # if os.path.exists(output_filename):
            #     continue
            # plt.clf()
            # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(figsize_width/my_dpi, figsize_height/my_dpi), dpi=my_dpi)

            # Plot FFT data
            fft      = np.load(source_dir_fft + th_file)
            th       = np.load(source_dir_th + th_file)
            # band power
            band_power_db_1[counter-1, 0] = band_power(th, fft, band_start=0.1, band_end=1)
            band_power_db_2[counter-1, 0] = band_power(th, fft, band_start=1, band_end=10)
            band_power_db_3[counter-1, 0] = band_power(th, fft, band_start=10, band_end=20)

            # xlim_fft = 50.0
            # ylim_fft = 25.0
            # ax1.set_xlabel("Frequency (Hz)")
            # ax1.set_ylabel("Intersity")
            # ax1.set_xlim([0, xlim_fft])
            # ax1.set_ylim([0, ylim_fft])
            # ax1.xaxis.set_major_locator(MultipleLocator(5))
            # ax1.xaxis.set_minor_locator(MultipleLocator(1))
            # ax1.yaxis.set_major_locator(MultipleLocator(5))
            # ax1.yaxis.set_minor_locator(MultipleLocator(1))
            # ax1.grid(True, which='major', color="gray",            linewidth=0.5, linestyle="-" )
            # ax1.grid(True, which='minor', color="gray", alpha=0.5, linewidth=0.5, linestyle="--")
            # ax2x_fft = ax1.secondary_xaxis("top")
            # ax2y_fft = ax1.secondary_yaxis("right")
            # ax1.set_xlim([0, xlim_fft])
            # ax2y_fft.set_ylim([0, ylim_fft])
            # ax2x_fft.xaxis.set_major_locator(MultipleLocator(5))
            # ax2x_fft.xaxis.set_minor_locator(MultipleLocator(1))
            # ax2y_fft.yaxis.set_major_locator(MultipleLocator(5))
            # ax2y_fft.yaxis.set_minor_locator(MultipleLocator(1))
            # ax1.title.set_text("Frequency")
            # #plt.grid()
            # average_fft = moving_average(fft[:,1], int(len(fft)/100))
            # ax1.plot(fft[:,0], fft[:,1]   , color="blue", linewidth=linewidth)
            # ax1.plot(fft[:,0], average_fft, color="yellow", linewidth=linewidth)
            # plt.tight_layout()
            # plt_savefig(output_filename, my_dpi)
        # save csv
        band_power_db_1 = zscore(band_power_db_1, axis=0)
        band_power_db_2 = zscore(band_power_db_2, axis=0)
        band_power_db_3 = zscore(band_power_db_3, axis=0)
        # print(band_power_db_1)
        
        # np.savetxt(output_dir + "band_power.csv", band_power_db_1, delimiter=",")
        plt.figure(1)
        plt.plot(band_power_db_1)
        plt.savefig(output_dir + "band_power_"+"0.1_1"+".png")
        plt.figure(2)
        plt.plot(band_power_db_2)
        plt.savefig(output_dir + "band_power_"+"1_10"+".png")
        plt.figure(3)
        plt.plot(band_power_db_3)
        plt.savefig(output_dir + "band_power_"+"10_20"+".png")
        plt.show()
        
    else:
        print("    python3 " + sys.argv[0] + " <source_dir_th> <source_dir_fft> <output_dir> <time_length (seconds)>")
        print("Ex. python3 " + sys.argv[0] + " /home/aries/Working/Infrasound/db/th_3600/ /home/aries/Working/Infrasound/db/fft_3600/ /home/aries/Working/Infrasound/images/th_fft_3600/ 3600")
        os.sys.exit(0)

        # python 6_band_power.py D:\Infrasound\db\th+fft\syk01_th3600 D:\Infrasound\db\th+fft\syk01_fft3600 D:\Infrasound\db\band_power\syk01 3600