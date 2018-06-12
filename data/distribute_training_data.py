import numpy as np

import os.path
import shutil
import pandas as pd

from configure import environment


def distribute_data(tce_list, class_list, data_type):
    print('\n')
    print("Distributing %s data with %d classes.\n" % (data_type, len(class_list)))
    print(class_list)

    list_length = len(tce_list)

    data_type_folder = os.path.join(environment.TRAINING_FOLDER, data_type)

    if os.path.exists(data_type_folder):
        # Folder already exist, delete it first and then create a new folder
        shutil.rmtree(data_type_folder)
    os.mkdir(data_type_folder)

    for i in range(list_length):
        tce_data = tce_list.iloc[i]

        kep_id = "%.9d" % int(tce_data.kepid)

        kepid_dir = os.path.join(environment.KEPLER_DATA_FOLDER, kep_id[0:4], kep_id)
        file_name = os.path.join(kepid_dir, "{0:09d}_plnt_num-{1:02d}_tce.record".format(int(tce_data.kepid),
                                                                                         tce_data.tce_plnt_num))

        if not os.path.isfile(file_name):
            print("File {} doesn't exist. Skipped.".format(file_name))
            break

        # Using different sub-folder for different labeled files
        # 'Train/0_PC'
        # 'Train/1_NON_PC' (could be 'AFP' from original label)
        # ...
        # 'Validation/1_NON_PC' (could be 'NTP' from original label)
        # ...
        #
        # To simplify the training, just merge all other non-plant
        # candidates to be as "NON_PC"
        if tce_data.av_training_set == 'PC':
            label_folder = '0_PC'
        else:
            label_folder = '1_NON_PC'

        #dest_folder = os.path.join(environment.TRAINING_FOLDER, data_type,
        #                            '{0}_{1}'.format(label_encoded, tce_data.av_training_set))
        dest_folder = os.path.join(environment.TRAINING_FOLDER, data_type, label_folder)
        if not os.path.exists(dest_folder):
            os.mkdir(dest_folder)

        print("Copying {0} to {1}, {2:5d} of {3}".format(file_name, dest_folder, i, list_length))
        shutil.copy2(file_name, dest_folder)


def main():
    # =========================================================================
    # Prepare training/validation/test data set

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

    # Randomly shuffle the TCE table.
    np.random.seed(123)
    tce_table = tce_table.iloc[np.random.permutation(num_tces)]
    # tf.logging.info("Randomly shuffled TCEs.")
    print("Randomly shuffled TCEs.")
    # print(tce_table)

    # Partition the TCE table as follows:
    #   train_tces = 80% of TCEs
    #   val_tces = 10% of TCEs (for validation during training)
    #   test_tces = 10% of TCEs (for final evaluation)
    train_cutoff = int(0.80 * num_tces)
    val_cutoff = int(0.90 * num_tces)

    train_tces = tce_table[0:train_cutoff]
    val_tces = tce_table[train_cutoff:val_cutoff]
    test_tces = tce_table[val_cutoff:]
    print("Partitioned {} TCEs into training ({}), validation ({}) and test ({})".format(
        num_tces, len(train_tces), len(val_tces), len(test_tces)))

    # The sub-folder name for 'Train'/'Validation'/'Test' sets are
    # intended to be as hardcoded in this project. No change.
    distribute_data(train_tces, environment.ALLOWED_LABELS, 'Train')
    distribute_data(val_tces, environment.ALLOWED_LABELS, 'Validation')
    distribute_data(test_tces, environment.ALLOWED_LABELS, 'Test')


if __name__ == "__main__":
    main()
