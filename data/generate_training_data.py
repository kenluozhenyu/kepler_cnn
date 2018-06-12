import numpy as np

import matplotlib.pyplot as plt
import threading
import math
import time
import os.path
import glob
import random
import pandas as pd
import pickle

from light_curve_util import kepler_io
from light_curve_util import median_filter
from light_curve_util import util
from third_party import kepler_spline

from data import preprocess
from data import training_data_io
from configure import environment

kepid_to_file_list = {}
total_num_kepids = 0

mutex = threading.Lock()

# Provide an option to us multi-threading to process the data
# But as the operations are mainly on disk I/O, using# multi-
# threading may not improve that much
#
# Change this value to "True" to use multi-threading
USING_MULTI_THREAD = True

NUM_OF_THREADS = 8


class myThread(threading.Thread):
    def __init__(self, threadID, name, tce_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.tce_list = tce_list
        self.num_tce = len(tce_list)

    def run(self):
        print("Start thread: " + self.name)
        # print_time(self.name, self.counter, 5)

        for i in range(self.num_tce):
            # selected_tce = self.tce_list.iloc[i]
            selected_tce = self.tce_list[i]

            print("{0} is converting Kepler ID = {1:09d}, plnt_num ={2:2d}. ({3:5d} of {4})"
                  .format(self.name, int(selected_tce.kepid), selected_tce.tce_plnt_num, i+1, self.num_tce))
            training_data_io.tce_global_view_local_view_to_file(selected_tce)

        print("\nTotally {} TCE converted".format(self.num_tce))
        print("Exit thread: {}\n".format(self.name))


def main():
    # Read CSV file of Kepler KOIs.
    tce_table = pd.read_csv(environment.KEPLER_CSV_FILE, index_col="loc_rowid", comment="#")
    tce_table["tce_duration"] /= 24  # Convert hours to days.
    # tf.logging.info("Read TCE CSV file with %d rows.", len(tce_table))
    print("Read TCE CSV file with {} rows.".format(len(tce_table)))
    # print(tce_table)

    # Filter TCE table to allowed labels.
    allowed_tces = tce_table[environment.LABEL_COLUMN].apply(lambda l: l in environment.ALLOWED_LABELS)
    tce_table = tce_table[allowed_tces]
    num_tces = len(tce_table)
    # tf.logging.info("Filtered to %d TCEs with labels in %s.", num_tces,
    #                 list(_ALLOWED_LABELS))
    print("Filtered to {} TCEs with labels in {}.".format(num_tces, list(environment.ALLOWED_LABELS)))
    print('Removed the "UNK (unknown) records"')
    # print(tce_table)

    print("Total TCE: {}".format(num_tces))

    if USING_MULTI_THREAD:
        size_per_sublist = math.ceil(num_tces / NUM_OF_THREADS)

        tce_list_set = []

        for i in range(NUM_OF_THREADS):
            tce_list_set.append([])

        for i in range(num_tces):
            list_index = math.trunc(i / size_per_sublist)
            if list_index > NUM_OF_THREADS - 1:
                list_index = NUM_OF_THREADS - 1
            tce_list_set[list_index].append(tce_table.iloc[i])

        # print(tce_list_set)

        for i in range(NUM_OF_THREADS):
            print(len(tce_list_set[i]))

        thread_set = [myThread(i + 1, "Thread-{0:03d}".format(i + 1), tce_list_set[i]) for i in range(NUM_OF_THREADS)]

        for thread_process in thread_set:
            thread_process.start()

        for thread_process in thread_set:
            thread_process.join()

        print("\nExit the main process")
    else:
         for i in range(num_tces):
            selected_tce = tce_table.iloc[i]

            print("Converting Kepler ID = {0:09d}, plnt_num ={1:2d}. ({2:5d} of {3})"
                  .format(int(selected_tce.kepid), selected_tce.tce_plnt_num, i, num_tces))
            training_data_io.tce_global_view_local_view_to_file(selected_tce)


if __name__ == "__main__":
    main()

