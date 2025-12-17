from copy import deepcopy
from datetime import datetime
#from scipy.fft import fft, ifft, rfft, fftfreq, rfftfreq
from scipy.fft import rfft, rfftfreq
import math
import numpy as np
import obspy
import os
import pickle



time_one_day_seconds = 86400.0



class mseed_record_info():

    def __init__(self, filename):
        extension = filename[filename.rfind("."):].lower()
        if (extension == ".pickle"):
            # Pickle file (binary)
            with open(filename, "rb") as f:
                self.__dict__ = pickle.load(f).__dict__
        else:
            self._filename = filename
            self.load_data()

    def save(self, output_file_full_name):
        if self._valid:
            output_folder = output_file_full_name[:output_file_full_name.rfind(os.sep)+1]
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            with open(output_file_full_name, "wb") as f:
                pickle.dump(self, f)

    def load_data(self):
        temp_data      = obspy.read(self._filename)[0]
        temp_stats     = temp_data.stats
        self._data     = temp_data.data
        self._max_data = np.max(self._data)
        self._min_data = np.min(self._data)
        if (self._min_data == self._max_data):
            self._valid = False
            return False
        elif ((self._min_data == 0.0) and (self._max_data == 0.0)):
            self._valid = False
            return False
        else:
            self._valid = True
        time_difference                  = 3600*8
        time_one_day_seconds             = 3600*24
        data_evaluation_ratio            = 0.001
        data_evaluation_criterion        = 0.02
        #print(self._data)
        self._site                       = temp_stats.station.lower()
        self._dt                         = temp_stats.delta
        self._data_num                   = temp_stats.npts
        self._starttime_timestamp        = temp_stats.starttime.timestamp
        self._endtime_timestamp          = temp_stats.endtime.timestamp
        del temp_stats
        del temp_data
        #self._data_num                   = len(self._data)
        self._starttime_timestamp        = np.round(self._starttime_timestamp - time_difference, 6)
        self._endtime_timestamp          = np.round(self._endtime_timestamp   - time_difference, 6)
        self._first_timestamp_of_the_day = np.round(((self._starttime_timestamp + time_difference) // time_one_day_seconds) * time_one_day_seconds, 6) - time_difference
        self._last_timestamp_of_the_day  = self._first_timestamp_of_the_day + time_one_day_seconds
        self._nonzero_data_index_start   = 0
        self._nonzero_data_index_end     = self._data_num - 1
        real_data_num                   = self._data_num
        if (self._endtime_timestamp > self._last_timestamp_of_the_day):
            time_today_part    = self._last_timestamp_of_the_day - self._starttime_timestamp
            time_tomorrow_part = self._endtime_timestamp         - self._last_timestamp_of_the_day
            if (time_today_part >= time_tomorrow_part):
                data_num_extra = int(np.ceil((self._endtime_timestamp - self._last_timestamp_of_the_day) / self._dt))
                self._nonzero_data_index_end -= data_num_extra
                #self._real_endtime_timestamp  = np.round(self._endtime_timestamp - data_num_extra * self._dt, 6)
            else:
                first_timestamp_of_the_day += time_one_day_seconds
                self._last_timestamp_of_the_day  += time_one_day_seconds
                data_num_extra = int(np.ceil((first_timestamp_of_the_day - self._starttime_timestamp) / self._dt))
                self._nonzero_data_index_start += data_num_extra
                #self._real_starttime_timestamp = np.round(self._starttime_timestamp + data_num_extra * self._dt, 6)
                time_today_part    = self._last_timestamp_of_the_day - self._starttime_timestamp
                time_tomorrow_part = self._endtime_timestamp         - self._last_timestamp_of_the_day
                if (self._endtime_timestamp > self._last_timestamp_of_the_day):
                    if (time_today_part >= time_tomorrow_part):
                        data_num_extra= int(np.ceil((self._endtime_timestamp - self._last_timestamp_of_the_day) / self._dt))
                        self._nonzero_data_index_end -= data_num_extra
                        #self._real_endtime_timestamp  = np.round(self._endtime_timestamp - data_num_extra * self._dt, 6)
        self._real_starttime_timestamp = self._starttime_timestamp
        for i in range(self._nonzero_data_index_start, self._nonzero_data_index_end-1, 1):
            if (self._data[i] != 0.0):
                self._nonzero_data_index_start = i
                self._real_starttime_timestamp = np.round(self._starttime_timestamp + i * self._dt, 6)
                break
        self._real_endtime_timestamp = self._endtime_timestamp
        for i in range(self._nonzero_data_index_end, self._nonzero_data_index_start-1, -1):
            if (self._data[i] != 0.0):
                self._nonzero_data_index_end = i
                self._real_endtime_timestamp = np.round(self._starttime_timestamp + i * self._dt, 6)
                break
        self._real_data_num   = int(self._nonzero_data_index_end - self._nonzero_data_index_start + 1)
        self._time_series     = np.round(self._real_starttime_timestamp + self._dt * np.arange(0, self._real_data_num, 1) - self._first_timestamp_of_the_day, 6)
        self._real_mean_value = np.mean(self._data[self._nonzero_data_index_start:self._nonzero_data_index_end+1])
        self._real_std_value  = np.std (self._data[self._nonzero_data_index_start:self._nonzero_data_index_end+1])
        self._real_min_value  = np.min (self._data[self._nonzero_data_index_start:self._nonzero_data_index_end+1])
        self._real_max_value  = np.max (self._data[self._nonzero_data_index_start:self._nonzero_data_index_end+1])
        self._period_ratio    = ((self._real_data_num - 1) * self._dt / time_one_day_seconds)
        data_temp             = deepcopy(self._data[self._nonzero_data_index_start:self._nonzero_data_index_end+1])
        data_temp.sort()
        num_evaluation_data = int(data_evaluation_ratio * self._real_data_num)
        if ((np.abs((data_temp[num_evaluation_data-1]-data_temp[0])/data_temp[0])<=data_evaluation_criterion)) or ((np.abs((data_temp[-1]-data_temp[-num_evaluation_data])/data_temp[-1])<=data_evaluation_criterion)):
            self._valid = False
        else:
            if (self._real_min_value == self._real_max_value):
                self._valid = False
            else:
                self._valid = True
        del data_temp

    def mean_value(self):
        if self._valid:
            return self._real_mean_value
        else:
            return 0

    def std_value(self):
        if self._valid:
            return self._real_std_value
        else:
            return 0

    def min_value(self):
        return self._real_min_value

    def max_value(self):
        return self._real_max_value

    def valid(self):
        return self._valid

    def exceed_limit(self, threshold, timestamp_start=None, timestamp_end=None):
        if ((timestamp_start == None) or (timestamp_start < self._starttime_timestamp)):
            index_start = 0
        elif (timestamp_start > self._endtime_timestamp):
            index_start = self._real_data_num - 1
        else:
            index_start = int(round((timestamp_start-self._starttime_timestamp)/self._dt, 0))
        if ((timestamp_end == None) or (timestamp_end > self._endtime_timestamp)):
            index_end = self._real_data_num - 1
        elif (timestamp_end < self._starttime_timestamp):
            index_end = 0
        else:
            index_end = int(round((timestamp_end-self._starttime_timestamp)/self._dt, 0))
        if (index_start != index_end):
            if ((abs(np.min(self._data[index_start:index_end+1])) >= abs(threshold)) or (abs(np.max(self._data[index_start:index_end+1])) >= abs(threshold))):
                return True
            else:
                return False
        else:
            return True

    def fft(self, start_index=0, end_index=0):
        if (end_index == 0):
            end_index = self._real_data_num - 1
        if ((start_index+1) < end_index):
            data_num = end_index - start_index + 1
            xf_abs   = rfftfreq(data_num, self._dt)
            yf_abs   = self._dt / time_one_day_seconds * np.abs(rfft(self._data[start_index:end_index+1]-self._real_mean_value))
            num_plot = min(len(xf_abs), len(yf_abs))
            return np.vstack((xf_abs[:num_plot], yf_abs[:num_plot])).T
        else:
            return None

    def piece_index(self, piece_start_timestamp, piece_end_timestamp):
        if (piece_start_timestamp >= self._real_starttime_timestamp):
            if (piece_start_timestamp > self._real_endtime_timestamp):
                start_index = -1
                end_index   = -1
            else:
                start_index = math.ceil((piece_start_timestamp - self._real_starttime_timestamp) / self._dt)
                if (piece_end_timestamp <= self._real_endtime_timestamp):
                    end_index = math.floor((piece_end_timestamp - self._real_starttime_timestamp) / self._dt)
                else:
                    end_index = self._nonzero_data_index_end
        else:
            if (piece_end_timestamp < self._real_starttime_timestamp):
                start_index = -1
                end_index   = -1
            else:
                start_index = 0
                end_index   = math.floor((piece_end_timestamp - self._real_starttime_timestamp) / self._dt)
        return [int(start_index), int(end_index)]

    def clean_data(self):
        del self.time_series
        del self._data
        del self.xf_abs
        del self.yf_abs

    def starttime_timestamp(self):
        return datetime.fromtimestamp(self._starttime_timestamp).strftime("%Y.%m.%d")

    def show(self):
        #self.load_data()
        print("File                      : " + self._filename)
        #print("Time                      : ", self.time_series)
        print("Site                      : " + self._site)
        print("dt                        : " + str(self._dt))
        print("First timestamp of the day: " + str(self._first_timestamp_of_the_day) + "\t(", datetime.fromtimestamp(self._first_timestamp_of_the_day), ")")
        print("Last timestamp of the day : " + str(self._last_timestamp_of_the_day ) + "\t(", datetime.fromtimestamp(self._last_timestamp_of_the_day ), ")")
        print("Start timestamp           : " + str(self._starttime_timestamp        ) + "\t(", datetime.fromtimestamp(self._starttime_timestamp        ), ")")
        print("End timestamp             : " + str(self._endtime_timestamp          ) + "\t(", datetime.fromtimestamp(self._endtime_timestamp          ), ")")
        print("Real start timestamp      : " + str(self._real_starttime_timestamp   ) + "\t(", datetime.fromtimestamp(self._real_starttime_timestamp   ), ")")
        print("Real end timestamp        : " + str(self._real_endtime_timestamp     ) + "\t(", datetime.fromtimestamp(self._real_endtime_timestamp     ), ")")
        print("Data index                : 0~" + str(self._data_num-1))
        print("Data index (real)         : " + str(self._nonzero_data_index_start) + "~" + str(self._nonzero_data_index_end))
        print("Number of records         : " + str(self._data_num))
        print("Number of records (real)  : " + str(self._real_data_num))
        print("Data                      : ", self._data)
        print("Data (real)               : ", self._data[self._nonzero_data_index_start:self._nonzero_data_index_end+1])
        print("Extreme value             : " + str(self._min_data) + " ~ " + str(self._max_data))
        print("Mean value                : " + str(self.mean()))
        print("STD value                 : " + str(self.std()))
        print("Day period                : " + str(self._period_ratio) + " day")
        if self._valid:
            print("Valid record              : Yes")
        else:
            print("Valid record              : No")
