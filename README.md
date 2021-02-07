# Racial_Classification_XAI_Model

### Introduction
* This is a project about visualizing Convolutional Neural Network (CNN) with XAI techniques such as Grad-cam and Integrated-Gradient. We used FairFace dataset for training. This dataset contains about 80000 training images and 10000 validation images. The dataset has the labeling of age range(9 categories), race(7 categories), and gender(2 categories). We developed a model structure that conbined the first 14 layers from resnet50 as pre-trained weights with our self-defined layers. We trained 3 models on each of the different classes using the same defined model structure except changing the number of outputs to match the number of categories from each class. Then, we applied XAI to visualize the learning process of the models. We want to examine the difference between the activation maps of different categories from a single class. Finally, we want to train a model with a "biased dataset" and compare its activation maps with the model that we trained previously(should be unbiased since the distributions of different categories have roughly the same number of training samples). 

##### config
* The parameters to run the scripts. Make sure to visit this file before running the code.

##### run.py
* script to train the model.

##### integrated_grad.py
* script to generate integrated-gradient activation maps from the model given images.

##### generate_plots.py
* script to generate the plots that inform the training progress and some statistics calculated with different metric scores.

##### src
* folder that contains the source code.

##### notebook
* still working...(ignore this for now).

##### mapping
* the mapping of categories to number.

##### models
* the trained models saved in .hdf5 format.

##### logs
* The training progress saved in .csv format.

##### visualizations
* contains plots generated using the generate_plots script.

### experimenting
* Contains development codes for our own use.

### How to run the code
1. please pull the my docker image: `michael459165/capstone2:new6` and run the code inside this container.
2. please go to the config file to change the parameters. This file has 4 sections, each corresponds to the parameters of a particular task.
3. visit "load_data" and "model_param" in the config file and customize your own parameters. Type `python run.py` to train your model.
4. visit the "integrated_grad" in the config file and customize your own parameters. Type `python integrated_grad.py` to generate the activation maps of your model.
5. visit the "generate_stats in the config file and customize your own parameters. Type `python generate_plots.py` to generate statistics and plots.

### Testing if your input is valid
1. After you customized your parameters for the "model_param", please type `python run.py test`. Check the returned message, if it says ""You are all set! Please type python run.py to train your model.", then type `python run.py` in the terminal.
2. After you customized your parameters for the "integrated_grad", please type `python integrated_grad.py test`. Check the returned message, if it says ""You are all set! Please type "python integrated_grad.py to generate the activation maps.", then type `python integrated_grad.py` in the terminal.


### Moving forward
* Add the grad-cam algorithm from Sudiksha's repo for generating grad-cam activation maps.
* Calculate the mean activation maps for each category of a particular class (e.g. the mean "White" activation map from race class).
* Train a model with a biased dataset.
* Implement face detction algorithm with dlib and resize the image to 224x224x3 that matches the input shape of our trained models.
* Develop back-end tool to connect the model with the Web App that will serve as our final product.

