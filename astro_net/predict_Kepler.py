# =============================================================================
# This procedure is to use the existing trained model to predict a TCE.
#
# We'll first look into the environment.KEPLER_DATA_FOLDER if the corresponding
# Kepler ID files have bee downloaded.
#
# If not downloaded yet, we'll download the files to
# environment.KEPLER_UNCLASSIFIED_DATA_FOLDER
#
# Then get the global view and the local view from the light-curve files and
# send them to the model to do prediction.
# =============================================================================
from keras.optimizers import Adam

import numpy as np
import os.path
import matplotlib.pyplot as plt

from astro_net import Kepler_cnn
from data import preprocess
from configure import environment
from download import Download_one_Kepler_ID

class tce_struct:
    kepid = 0
    tce_period = 0.0
    tce_time0bk = 0.0
    tce_duration = 0.0


def predict_by_kepler_tce(tce):
    X1, X2 = [], []

    model = Kepler_cnn.build_Kepler_CNN()

    # Set the metrics. Only use top k if there's a need.
    metrics = ['accuracy']
    if environment.NB_CLASSES >= 10:
        metrics.append('top_k_categorical_accuracy')

    # Now compile the network.
    optimizer = Adam(lr=1e-5, decay=1e-6)
    # optimizer = 'adadelta'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=metrics)

    # Load the existing trained model
    trained_model_filename = os.path.join(environment.KEPLER_TRAINED_MODEL_FOLDER, 'kepler-model-two-classes.h5')

    if os.path.isfile(trained_model_filename):
        model.load_weights(trained_model_filename)

    # =========================================================================
    # Get the light-curve files

    # First, look for the KEPLER_DATA_FOLDER, see if we already have that Kepler ID downloaded
    kepid_formatted = "{0:09d}".format(int(tce.kepid))  # Pad with zeros.

    download_folder = os.path.join(environment.KEPLER_DATA_FOLDER, kepid_formatted[0:4], kepid_formatted)

    from_unclassified_folder = False

    if not os.path.exists(download_folder):
        # We didn't have the light-curve files downloaded for this Kepler ID
        # Go to check if we have it downloaded to the KEPLER_UNCLASSIFIED_DATA_FOLDER
        download_folder = os.path.join(environment.DATA_FOR_PREDICTION_FOLDER,
                                       kepid_formatted[0:4], kepid_formatted)

        print("Target not in KEPLER_DATA_FOLDER. Look into the DATA_FOR_PREDICTION_FOLDER.")
        from_unclassified_folder = True

        if not os.path.exists(download_folder):
            # We don't have it downloaded to KEPLER_UNCLASSIFIED_DATA_FOLDER yet.
            # Do the download first
            print("Need to download to the DATA_FOR_PREDICTION_FOLDER first.")
            Download_one_Kepler_ID.download_one_kepler_id_files(tce.kepid)

    # =========================================================================
    # Get the global view and local view from the light-curve files

    if not from_unclassified_folder:
        time, flux = preprocess.read_and_process_light_curve(tce.kepid, environment.KEPLER_DATA_FOLDER, 0.75)
    else:
        time, flux = preprocess.read_and_process_light_curve(tce.kepid, environment.DATA_FOR_PREDICTION_FOLDER, 0.75)

    time, flux = preprocess.phase_fold_and_sort_light_curve(
        time, flux, tce.tce_period, tce.tce_time0bk)

    global_view = preprocess.global_view(time, flux, tce.tce_period)
    local_view = preprocess.local_view(time, flux, tce.tce_period, tce.tce_duration)

    # Change the dimension to fit for the model input shape
    global_view = np.reshape(global_view, (2001, 1))
    local_view = np.reshape(local_view, (201, 1))

    # =========================================================================
    # Save the global view and local view to a picture
    fig, axes = plt.subplots(1, 2, figsize=(10 * 2, 5), squeeze=False)
    axes[0][0].plot(global_view, ".")
    axes[0][0].set_title("Global view")
    axes[0][0].set_xlabel("Bucketized Time (days)")
    axes[0][0].set_ylabel("Normalized Flux")

    axes[0][1].plot(local_view, ".")
    axes[0][1].set_title("Local view")
    axes[0][1].set_xlabel("Bucketized Time (days)")
    axes[0][1].set_ylabel("Normalized Flux")

    fig.tight_layout()

    file_name = '{0}_period={1}_time0bk={2}_duration={3}.png'\
        .format(tce.kepid, tce.tce_period, tce.tce_time0bk, tce.tce_duration*24)
    file_name = os.path.join(environment.PREDICT_OUTPUT_FOLDER, file_name)

    if os.path.isfile(file_name):
        os.remove(file_name)

    fig.savefig(file_name, bbox_inches="tight")

    # =========================================================================
    # Do the prediction
    X1.append(global_view)
    X2.append(local_view)

    result_X = [np.array(X1), np.array(X2)]

    predict_result = model.predict(result_X, batch_size=1, verbose=0)

    predict_result_text = ""
    predict_result_index = 0

    if predict_result[0][0] > 0.5:
        predict_result_text = "PC (planet candidate)"
        predict_result_index = 0

    if predict_result[0][1] > 0.5:
        predict_result_text = "Not a planet candidate "
        predict_result_index = 1

    '''
    if predict_result[0][2] > 0.5:
        predict_result_text = "NTP (non-transiting phenomenon)"
        predict_result_index = 2
    '''

    print("\nKepler ID    = {:9d}".format(tce.kepid))
    print("tce_period   = {}".format(tce.tce_period))
    print("tce_time0bk  = {}".format(tce.tce_time0bk))

    # Print the duration in hours value
    print("tce_duration = {}".format(tce.tce_duration*24))
    # print("tce_duration = {}".format(tce.tce_duration))

    print("\n==> Predicted result = {0}".format(predict_result))

    # We are using the two-class category: ['0_PC', '1_NON_PC'] instead
    # print("==> Available labels: {}".format(environment.ALLOWED_LABELS))
    print("==> Available labels: {}".format(['0_PC', '1_NON_PC']))
    print("==> {0:.00%} possibility is a {1:s}".format(predict_result[0][predict_result_index], predict_result_text))

    # fig.show()


def main():
    tce = tce_struct()

    # tce.kepid = 11442793
    # tce.tce_period = 331.603
    # tce.tce_time0bk = 140.48
    # tce.tce_duration = 14.49

    tce.kepid = 11442793
    tce.tce_period = 14.44912
    tce.tce_time0bk = 2.2
    tce.tce_duration = 2.70408 # in hours, = 0.11267 days

    '''
    tce.kepid = 757450
    tce.tce_period = 8.88492
    tce.tce_time0bk = 134.452
    tce.tce_duration = 2.078  # in hours
    '''

    # Convert duration to days
    tce.tce_duration /= 24

    predict_by_kepler_tce(tce)


if __name__ == "__main__":
    main()
