This is my learning and re-write some code from this project:
https://github.com/tensorflow/models/tree/master/research/astronet

Main modifications:
1) Re-wrote the CNN model in Keras framework
   Code in astro_net/Keper_cnn.py
2) Re-wrote the download program
3) Re-wrote the record file generation, and a generator for the Keras model

How to use:
1) Check the configure/environment.py and make sure the corresponding folders
   are created.
2) If you just want to do a prediction, run the astro_net/predict_Kepler.py
   The trained model is stored in the astro_net/trained_model/xxx.h5
   You ma need to change line 48 for the file name as well
   
   The program will download the corresponding light-curve files to 
   environment.DATA_FOR_PREDICTION_FOLDER
   
   The prediction result is printed. The image for global view and local view
   will be saved in environment.PREDICT_OUTPUT_FOLDER
   
3) If you want to do the training yourself
   a) Get the q1_q17_dr24_tce.csv accoridng to the instruction from
      https://github.com/tensorflow/models/tree/master/research/astronet
      
      And put the file to environment.KEPLER_CSV_FILE
      
   b) Run these two scripts in the download/ folder:
      Download-Kepler-data-Step1-query-file-list.py
      Download-Kepler-data-Step2-get-files.py
      
      You'll need to check the folder name in line 34 (step1) and 41 (step2)
      Note: It will take about two days for the download.
      
   c) Run data/generate_training_data.py to convert the original data to
      the form that the CNN model can use.
      
      Note: This is very slow. It will take several days.
      
   d) Run data/distribute_training_data.py
      It will put the record files to Train/Validation/Test folders
   
   Run astron_net/train_Kepler_CNN.py to train the model
  
