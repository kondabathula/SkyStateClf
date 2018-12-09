﻿# SkyStateClf
Sky state classification using Resnet 34. There are two models in this repo. One of which classifies an input sky image to one of the three states *Cloudy*, *Partly_cloudy* and *Clear_sky*, in addition to these classes another model predicts *Mostly Cloudy* class also.

# Intro
Detecting sky condition is one of the important aspect in the study of atmospheric physics as the presence of clouds dictate atmospheric properties and sometimes they are reflection of some atmospheric properties. The present methods as far as I know are not universally applicable and are sometimes location specific. I have not seen a purely deep learning solution to this problem and stated with simple classification model. I am looking forward to developing segmentation model using W-Net architecture (it is an extended version of U-net segmentation architecture for unsupervised learning application) to mask regions of clouds in a given image which would be very interesting for the atmospheric community.

# Data source
The images to train the model are collected from Trans-European Network of Cameras for Observation of Noctilucent Clouds. The description of the camera network and observations made from the network are in [G. Baumgarten, M. Gerding, B. Kaifler, & N. Müller 2009. A Trans European Network of Cameras for Observation of Noctilucent Clouds From 37 ON to 69ON].

# Model and result
The pretrained [ResNet34](https://github.com/KaimingHe/deep-residual-networks) [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) model is loaded from tensorvision modules and last layer is modified to predict for three classifications. 50000 images were used to train the model and 8000 images are used to validate the results. The model achieved 97.6% accuracy on validation set and 98.2% accuracy on training set. A four state classification model is also developed uisng fastai library using ResNet34 architecture and achieved 96.03% accuracy on training set and 96.08% accuracy on validation set.

# Preparing dataset and training
As preparing the data is the major takes in machine learning applications, I will try to explain the steps I took to easily prepare a good dataset. (This might not be a new strategy or might not be applicable for other kinds of data)

I manually segregated around 1500 images each of classes completely clear sky, completely cloudy and partially/mostly cloudy images and trained the model using these images, later I ran this trained model on all the images available. Since the clear sky and cloudy images have less features to learn the model prediction was approximately 85% accurate. I used these prediction to further segregate 10000 images of each of the three classes (which took around half an hour), I repeated the training of the model and got around than 95% accuracy (keeping in mind that often it is very challenging even for human eye to classify clear sky condition to a cloudy condition, this result is very promising for me to go further). 
From the third class of images partly/mostly cloudy the above procedure is repeated to separate almost clear sky images, partly and mostly cloudy images. Though, almost clear sky images were accurately segregated the model was not very accurate between partly and mostly cloudy images. This can be understood by asking a question to our self, that how accurately and consistently the training dataset is defining the boundaries of partly cloudy and mostly cloudy images? To make my model simple, I decided to confine to three classes. One will be completely clear sky to very little/faint clouds, second class defines between faint clouds and mostly cloudy conditions and third class defines from mostly cloudy to completely cloudy. In this way, I have a bigger room for any wrong predictions in second class (just like one person may think ‘it’s partly cloudy’ and other person might think ‘it’s mostly cloudy’, the difference is subtle).
Finally, after segregate the dataset of 58000 images. The last layer of pretrained ResNet34 model is modified to output three class and trained only the last layer, this training exercise produced an accuracy of about 90% later, all the inner layers are also trained, and *97.2% accuracy* is achieved. You can see the training script in “skyClfPytorch_trainer.ipynb” file.

## Using FastAi library.
I went on to develop a four state (Clear sky, partly cloudy, mostly cloudy and cloudy) classifier using fastai library. I used [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) for this model achieved *96.03% accuracy*. The figure below depict the confusion matrix, the name confusion is missleading, however it provides a good representation of model performance.

<img src="Images/confusion matrix.JPG" width="50%">

# Usage
In “SkyClf_predict.ipynb” file change the image location to run on your image. Remember, you need Pytorch and Torchvision libraries to run this script. Both the libraries are available for GPU version and CPU version.
Hope this model is useful for your application without any further training. You can find the three state trained parameters in “ResNet34_parameters.pth” file and four state trained parameters using fastai library in “SkyStateClf_param_fastai.h5” file, while using four state classification remember to change the last layer of your model to give four outputs.
