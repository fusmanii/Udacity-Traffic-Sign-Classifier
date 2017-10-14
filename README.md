## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)
* [Jupyter] (http://jupyter.org/)
* [NumPy] (http://www.numpy.org/)
* [OpenCV] (https://opencv.org/)
* [Mathplotlib] (https://matplotlib.org/index.html)
* [Tensorflow] (https://www.tensorflow.org/)
* [SciPy] (https://www.scipy.org/)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```


#**Traffic Sign Recognition**

[//]: # (Image References)

[image1]: ./writeup_images/1.png "Visualization"
[image2]: ./writeup_images/2a.png "Distribution"
[image3]: ./writeup_images/2b.png "Distribution"
[image4]: ./writeup_images/2c.png "Distribution"
[image5]: ./writeup_images/3.png "Grayscale"
[image6]: ./writeup_images/4a.png "Brightness"
[image7]: ./writeup_images/4b.png "Rotation and scale"
[image8]: ./writeup_images/4c.png "Translation"
[image9]: ./writeup_images/5.png "Augmentation"
[image10]: ./writeup_images/6.png "Distribution"
[image11]: ./writeup_images/7.png "Normalization"
[image12]: ./writeup_images/8.jpeg "Model"
[image13]: ./writeup_images/9a.jpg "Web image"
[image14]: ./writeup_images/9b.jpg "Web image"
[image15]: ./writeup_images/9c.jpg "Web image"
[image16]: ./writeup_images/9d.jpg "Web image"
[image17]: ./writeup_images/9e.jpg "Web image"
[image18]: ./writeup_images/9f.jpg "Web image"
[image19]: ./writeup_images/9g.jpg "Web image"
[image20]: ./writeup_images/9h.jpg "Web image"
[image21]: ./writeup_images/10.png "Predictions"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup


You're reading it! and here is a link to my [project code](https://github.com/fusmanii/Udacity-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic. You can find the code on the IPython file liked above.

signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

##### 2a. Here is an exploratory visualization of the data set. 

This is sample of the image set:
![alt text][image1]

This is what the image distribution look like:

![alt text][image2]
![alt text][image3]
![alt text][image4]
 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The data set was first converted to grayscale:
![grayscale][image5]

Some image classes had way fewer samples than others, so to make the sample more evenly distributed the data was extended.

Random images were selected from each class, then brightness, translation, rotation and scale augmentations were applyed to the image. The reason behind the augmentation was to avoid overfitting.

Will show all image augmentation on a sample image:
##### Brightness
![Brightness][image6]

##### Rotation and scale
![Rotation and scale][image7]

##### Translation
![Translation][image8]

Here is an example of an original image and an augmented image:
![Augmentation][image9]

This is what the training data looks after the augmentation:

![Distribution][image10]

The pixel average of all the train images was 81.9549565999. It was normalized to 0.364244251555. This is what the images look like after the normalization:
![Normalization][image11]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model that was proposed on Sermanet/LeCunn traffic sign classification journal article was implemented.
![Model][image12]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					| 
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400
| RELU					|
| Concat Layer 2 & Layer 3 | output 1x1x800
| Dropout | drop rate 0.5
| Fully connected		| output 43       									|
| Softmax				|  											|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The following are the hyperparameters used during training:

* Epochs: 60
* Batch size: 128
* Learning rate: 0.0009
* Keep probability: 0.5
* Mean for weights: 0
* Sigma for weights: 0.1

The learning was the same as the LaNet lab. The AdamOptimizer was used for the learning and the data set was shuffled after every epoch.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 1.00
* validation set accuracy of 0.969
* test set accuracy of 0.950

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![Web image][image13] ![Web image][image14] ![Web image][image15] 
![Web image][image16] ![Web image][image17] ![Web image][image18]
![Web image][image19] ![Web image][image20]

All the images we some what easy to clasify. The images were resized to 48x48 so few images got a bit distorted which might cause some difficulties.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)      		| Speed limit (60km/h)  									| 
| Right-of-way at the next intersection      			| Right-of-way at the next intersection  										|
| Road work					| Road	work									|
| Priority road	      		| Priority road					 				|
| General caution			| General caution     
| Keep right	| Keep right
| Roundabout mandatory | Roundabout mandatory
| Speed limit (30km/h) | Speed limit (30km/h) 


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 100%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The following is the image prediction with softwax probabilities:
![Preditions][image21]
