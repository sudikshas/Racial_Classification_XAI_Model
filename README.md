# Racial_Classification_XAI_Model

Website: https://michael4706.github.io/XAI_Website/

### Introduction
This project is about visualizing Convolutional Neural Network (CNN) with XAI techniques: Grad-cam and Integrated-Gradient. We used the FairFace dataset to train our models. This dataset contains about 80000+ training images and 10000+ validation images. The dataset contains three different categories(labels): age range(9 classes), race(7 classes), and gender(2 classes). We implemented a model that combined the first 14 layers from resnet50 as pre-trained layers with our self-defined layers. We trained three models on each of the different categories using the same model structure except changing the number of outputs from the final layer to match each category's number of classes. Then, we applied XAI to visualize models' decision-making with heatmaps. We want to examine what features or regions the models focus on given an image. Also, we are interested in comparing the heatmaps generated by the biased and unbiased models. The FairFace Dataset has an equal distribution of race. Therefore, we created a dataset with an unequal distribution of race and trained a biased model with this dataset.

##### config
* The parameters to run the scripts. Make sure to visit this file before running the code.

##### run.py
* script to train the model, run integrated-gradients, and calculate the statistics for the model.

##### src
* folder that contains the source code.

##### notebook
* Contains development codes for our own use.

##### mapping
* the mapping of the class to number stored in .json file.

##### models
* the trained models saved in .hdf5 format.

##### logs
* The training progress saved in .csv format.

##### visualizations
* contains statistics and integrated-gradient visualization samples.

##### test_data
* Contains sample data to run just for the purpose of demo.

### How to run the code
1. please pull the my docker image: `michael459165/capstone2:new8` and run the code inside this container.
2. please go to the config file to change the parameters. This file has 4 sections, each corresponds to the parameters of a particular task.
3. visit "load_data" and "model_param" in the config file and customize your own parameters. Type `python run.py train_model` to train your model.
4. visit the "integrated_grad" in the config file and customize your own parameters. Type `python run.py integrated_grad` to generate the activation maps of your model.
5. visit the "generate_stats in the config file and customize your own parameters. Type `python run.py generate_plots` to generate statistics and plots.

### Testing the script
1. `python run.py test` will train a model with a very small dataset. This just serves as a simple demo to run
2. `python integrated_grad.py test` will create a integrated-grad pictures in the test_data/integrated_grad folder


### Moving forward
* Add the grad-cam algorithm from Sudiksha's repo for generating grad-cam activation maps.
* Calculate the mean activation maps for each category of a particular class (e.g. the mean "White" activation map from race class).
* Train a model with a biased dataset.
* Implement face detction algorithm with dlib and resize the image to 224x224x3 that matches the input shape of our trained models.
* Develop back-end tool to connect the model with the Web App that will serve as our final product.

### Reference
[1]Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE international conference on computer vision. 2017.

[2]Grad-CAM implementation in Keras[Source code]. https://github.com/jacobgil/keras-grad-cam.

[3]Sundararajan, Mukund, Ankur Taly, and Qiqi Yan. "Axiomatic attribution for deep networks." International Conference on Machine Learning. PMLR, 2017.

[4]Integrated Gradients[Source code]. https://github.com/hiranumn/IntegratedGradients.

[5]@inproceedings{karkkainenfairface,
      title={FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation},
      author={Karkkainen, Kimmo and Joo, Jungseock},
      booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
      year={2021},
      pages={1548--1558}
    }

[6] FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age[Source code]. https://github.com/dchen236/FairFace.
