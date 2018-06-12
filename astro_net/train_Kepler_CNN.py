from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.utils import to_categorical

import numpy as np
import os.path
import random
import threading
import time
import os.path

from astro_net import Kepler_cnn
from configure import environment
from data import training_data_io

import tensorflow as tf
from keras import backend as K

# This is for environment has tensorflow-gpu installed but
# want to explicitly specify GPU or CPU is to be used.
# Sometimes model would be too complex that exceed the memory
# limit of the GPU so we may have to go for CPU
GPU = True
# GPU = False

# Could change below according to your h/w environment
if GPU:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 1
    num_GPU = 0

num_cores = 24

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
session = tf.Session(config=config)
K.set_session(session)

# Training parameters
BATCH_SIZE = 32
NB_EPOCH = 200

# How many times allowed when val_loss didn't decrease
TRAIN_patience = 30

TRAINING_SAMPLE_SIZE = 12600
VALIDATION_SAMPLE_SIZE = 1600

# If False, the just load the existing trained model and do the
# prediction with the test set
DO_TRAINING = True


# =============================================================================
# Define the data generator

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen


@threadsafe_generator
def tce_generator(data_folder, data_type, batch_size):
    # Training data folders organized in this way:
    #
    # 'Train/0_PC'
    # 'Train/1_NON_PC'
    # 'Validation/0_PC'
    # 'Validation/1_NON_PC'
    # 'Test/0_PC'
    # 'Test/1_NON_PC'
    #
    # The 'data_folder' parameter will point to '.../Train' or
    # '.../Validation' or '.../Test'

    class_list = os.listdir(data_folder)
    print('\n')
    print("Creating %s generator with %d classes." % (data_type, len(class_list)))
    print(class_list)

    while 1:
        X1, X2, y = [], [], []

        # Generate batch_size samples.
        for _ in range(batch_size):
            # Reset to be safe.
            #sequence = None
            # result_X = []

            # Get a random class. '0_PC' or '1_NON_PC'
            selected_class = random.choice(class_list)
            class_folder = os.path.join(data_folder, selected_class)

            # Get all samples' folder name.
            sample_list = os.listdir(class_folder)

            # Get a random sample.
            # The file name doesn't contain path info, just file name
            record_file = random.choice(sample_list)

            dest_file = os.path.join(class_folder, record_file)

            global_view, local_view = training_data_io.read_tce_global_view_local_view_from_file(dest_file)

            # Upon saving the data to file I have already reshaped them.
            # But looks like I still have to reshape them again after
            # reading from file.
            global_view = np.reshape(global_view, (2001, 1))
            local_view = np.reshape(local_view, (201, 1))

            # X.append(result_X)
            X1.append(global_view)
            X2.append(local_view)

            # Get the class index
            label_encoded = class_list.index(selected_class)
            label_hot = to_categorical(label_encoded, len(class_list))

            y.append(label_hot)

        yield [np.array(X1), np.array(X2)], np.array(y)


def main():
    # =========================================================================
    # Get the CNN model and do the training
    # or load the trained model

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

    if DO_TRAINING:
        # Helper: Save the model.
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(environment.KEPLER_TRAINED_MODEL_FOLDER, 'checkpoints',
                                  'CNN-train.{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            save_best_only=True)

        # Helper: TensorBoard
        tb = TensorBoard(log_dir=os.path.join(environment.KEPLER_TRAINED_MODEL_FOLDER, 'logs', 'CNN'))

        # Helper: Stop when we stop learning.
        early_stopper = EarlyStopping(patience=TRAIN_patience)

        # Helper: Save results.
        timestamp = time.time()
        csv_logger = CSVLogger(os.path.join(environment.KEPLER_TRAINED_MODEL_FOLDER, 'logs', 'CNN-training-' + \
                                            str(timestamp) + '.log'))

        # Get samples per epoch.
        steps_per_epoch = TRAINING_SAMPLE_SIZE / BATCH_SIZE

        # Get generators.
        # The sub-folder name for 'Train'/'Validation'/'Test' sets are
        # intended to be as hardcoded in this project. No change.

        train_set_folder = os.path.join(environment.TRAINING_FOLDER, 'Train')
        validation_set_folder = os.path.join(environment.TRAINING_FOLDER, 'Validation')

        train_generator = tce_generator(train_set_folder, 'Train', BATCH_SIZE)
        val_generator = tce_generator(validation_set_folder, 'Validation', BATCH_SIZE)

        # Perform the training now!
        model.fit_generator(
                    generator=train_generator,
                    steps_per_epoch=steps_per_epoch,
                    epochs=NB_EPOCH,
                    verbose=1,
                    callbacks=[tb, early_stopper, csv_logger, checkpointer],
                    validation_data=val_generator,
                    validation_steps=VALIDATION_SAMPLE_SIZE/BATCH_SIZE,
                    workers=24)

        trained_model_filename = os.path.join(environment.KEPLER_TRAINED_MODEL_FOLDER, 'kepler-model-two-classes.h5')
        model.save_weights(trained_model_filename)
    else:
        # Load the existing trained model
        trained_model_filename = os.path.join(environment.KEPLER_TRAINED_MODEL_FOLDER, 'kepler-model-two-classes.h5')

        if os.path.isfile(trained_model_filename):
            model.load_weights(trained_model_filename)

    # =========================================================================
    # Use the test set to verify the accuracy
    test_set_folder = os.path.join(environment.TRAINING_FOLDER, 'Test')
    class_list = os.listdir(test_set_folder)

    correct_count = 0
    test_set_size = 0

    for selected_class in class_list:
        print("\n")
        print(selected_class)
        print("\n")
        class_folder = os.path.join(test_set_folder, selected_class)

        # Get all samples' folder name.
        sample_list = os.listdir(class_folder)

        test_set_size += len(sample_list)

        for record_file in sample_list:
            result_X = []
            X1, X2= [], []

            dest_file = os.path.join(class_folder, record_file)
            global_view, local_view = training_data_io.read_tce_global_view_local_view_from_file(dest_file)

            global_view = np.reshape(global_view, (2001, 1))
            local_view = np.reshape(local_view, (201, 1))

            X1.append(global_view)
            X2.append(local_view)

            result_X = [np.array(X1), np.array(X2)]

            predict_correct = False

            predict_result = model.predict(result_X, batch_size=1, verbose=0)

            if selected_class == '0_PC':
                if predict_result[0][0] > 0.5:
                    predict_correct = True
            else:
                if predict_result[0][1] > 0.5:
                    predict_correct = True

            if predict_correct:
                correct_count += 1

            print("{0} ==> Predicted result = {1}, result = {2}"
                  .format(record_file, predict_result, predict_correct))

    print("\nTotal test sample = {0}. Correctly predicted = {1}. Accuracy = {2:.00%}"
          .format(test_set_size, correct_count, correct_count / test_set_size))


if __name__ == "__main__":
    main()
