# Deep-CNN based Multi-task Learning for Open-Set Recognition

This is official implimentation of the paper   [Deep-CNN based Multi-task Learning for Open-Set Recognition](https://arxiv.org/pdf/1903.03161.pdf)

# Installation
1. Install pytorch
2. Install Matlab
3. Clone this repository
  ```Shell
  git clone https://github.com/otkupjnoz/mlosr.git
  ```

# Data Setup
1. You can download the ood sets in mat format from here, 

https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/poza2_jh_edu/EeJ1RQQm425MsuFGfV5JO58BMU5Q9m2uzM_Nd3akC-MCLA?e=TXQtzR

2. For open-set experiments convert your datasets in matfiles (or modify the code to create your own dataloader.)
3. Make sure while saving in matlab you use '-v7.3'.
4. Create train, test and validation and save it by name 
   train_label.mat, train_data.mat, test_label.mat, test_data.mat, validation_data.mat, validation_label.mat etc.
5. Save all the datasets in the dataset/ folder

# Training
1. set up your data as described above
2. The code is running OOD experiments of the paper which uses pytorch dataloader
3. make sure you have your dataset mat files in datasets/data_set_name/
   make sure to add parameter.py file in master/parameters/data_set_name/
   make sure to add create following folders :
   ```Shell
   save_folder/models/data_set_name/mlosr
   save_folder/models/data_set_name/checkpoint
   save_folder/results/data_set_name/encoded_images
   ```
4. Use following command to run the code (make changes in the parameter file to run the code for different experiment)
   ```Shell
   sh run_train.sh
   ```
  
# Testing
1. Use following command
  ```Shell
  sh run_test.sh
  ```
2. open Matlab
3. run getResultsMLOSR.m file which will calculate and display the F-measure


