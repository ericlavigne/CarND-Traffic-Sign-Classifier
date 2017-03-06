## Project: Traffic Sign Recognition Using a Convolutional Neural Network
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
As part of Udacity's Self-Driving Car Nanodegree, I downloaded a collection of German traffic sign pictures and
created a neural network to classify those pictures into sign types such as "speed limit 30km/h" or "children crossing".
Despite some challenging pictures, such as very dimly lit signs, the neural network correctly classified 96% of images
in the test set.

Project steps included:
* Load the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report (this README file)

---

Data Set Summary and Exploration
---

There were three separate data sets: training, validation, and test. The idea
was to train the model using the training data set, use the validation data set 
to evaluate how well the model was doing as I tuned its parameters, then use the
test data set at the end for independent analysis of the model's accuracy.

Each data set contained a collection of 32x32 color images and corresponding
labels, indicating the type of sign. There were 43 types of signs, such as 
"speed limit 30 km/h" or "children crossing". The training, validation, and
test data sets contained 34799, 4410, and 12630 images respectively.
(See "Provide a Basic Summary" in the notebook.)

Upon visualizing one of each type of sign from the training set, I was
surprised to find that many of the images were very dim and/or blurry. Some
just looked like black squares rather than pictures of signs. I would need
to correct for the poor lighting as a preprocessing step.
(See "Exploratory Visualization" in the notebook.)

![examples of training data](https://raw.githubusercontent.com/ericlavigne/CarND-Traffic-Sign-Classifier-Project/master/figures/sign-examples.png)

Pre-processing
---

My main concern was the dim lighting of many of the training images. With that
in mind, my first preprocessing step was to correct the lighting. For each 
image, I applied a linear transformation to all pixel values such that the 
smallest pixel value would be 1 and the largest pixel value would be 253. This
transformation approximated the result of taking all the pictures in normal
lighting and was very effective in adjusting the dimly lit pictures.

I also tried converting the images from the typical Red/Green/Blue color
representation into Hue/Saturation/Value because I heard that other Udacity
students found this useful. The resulting pictures looked worse to me. Rather
than making a judgement call between these two representations, though, I
decided to include both with 6-channel input images.

The results of both transformation approaches are shown below.

![examples of pre-processed training data](https://raw.githubusercontent.com/ericlavigne/CarND-Traffic-Sign-Classifier-Project/master/figures/preprocessing.png)

Model Architecture
---

All convolutional and dense layers, except for the output layer, used batch
normalization, 50% dropout, and tanh activation. All dense layers, including
the output layer, used 0.01 L2 regularization. I used the Adam optimizer to
avoid tuning the learning rate.

Neural network architecture tends to be a tradeoff between training set
accuracy, generalization (fitting one training set leads to accurate predictions
on a second training set), and training speed. With access to a powerful GPU, I
could err on the side of producing a network with high capacity and
generalizability, and the GPU would minimize the resulting cost in speed.

For high training set accuracy, I added a lot of layers with a lot of depth
in each layer. This included 11 convolutional layers (generally a good choice
for image data), with filter depthsgradually increasing from 30 to
130 as later layers needed to represent  information about larger
portions of the image. This also included 4 dense
layers with depths of 130, 120, 120, and 43. The first three dense layers 
needed to be at least twice as large as the output layer to make up for the
50% dropout (half of all neurons would randomly be pulled out during training), 
and I increased that by another 50% to err on the side of high accuracy. I
also let the model train thoroughly: 200 epochs with 10000 samples per epoch.
The final training accuracy was 98%.

For generalization, I added 50% dropout to every layer, except the output
layer where it wouldn't make sense. I also added L2 normalization to the
dense layers. The 50% dropout, especially, represented an extreme amount of
regularization to prevent overfitting to the training data. The final accuracy
on the test set was 96%.

Despite the large network and heavy regularization, GPU accelleration allowed
training to complete in just 17 minutes.

| Layer          | Description                            | Output                           |
|:--------------:|:--------------------------------------:|:--------------------------------:|
| Input          | Original Image                         | 32x32x3 RGB image                |
| Pre-Processing | Normalize, w/ and w/o HSV conversion   | 32x32x6 RGB/HSV normalized image |
| Convolution    | 5x5, depth 30                          | 28x28x30                         |
| Convolution    | 5x5, depth 50                          | 24x24x50                         |
| Convolution    | 5x5, depth 60                          | 20x20x60                         |
| Convolution    | 5x5, depth 80                          | 16x16x80                         |
| Convolution    | 3x3, depth 100                         | 14x14x100                        |
| Convolution    | 3x3, depth 110                         | 12x12x110                        |
| Convolution    | 3x3, depth 130                         | 10x10x130                        |
| Convolution    | 3x3, depth 130                         | 8x8x130                          |
| Convolution    | 3x3, depth 130                         | 6x6x130                          |
| Convolution    | 3x3, depth 130                         | 4x4x130                          |
| Convolution    | 3x3, depth 130                         | 2x2x130                          |
| Flatten        | Transition from convolutional to dense | 520                              |
| Dense          | depth 130                              | 130                              |
| Dense          | depth 120                              | 120                              |
| Dense          | depth 120                              | 120                              |
| Dense          | depth 43                               | 43 (one for each sign type)      |

Testing the Model on New Images
---

I downloaded five additional German traffic sign images to further test the network.
These are all high-quality images with good lighting, high resolution, and no
obstructions. In other words, these should be very easy to classify. Despite 
this, I was surprised to find that the network correctly classified only 4 out
of 5 signs, much worse than the 96% accuracy that the network achieved on the
much more challenging data set from Udacity.

![5 downloaded German traffic signs](https://raw.githubusercontent.com/ericlavigne/CarND-Traffic-Sign-Classifier-Project/master/figures/downloaded.png)

The network correctly identified 4 of the signs with high confidence:
speed limit 30 km/h (98%), keep right (98%), children crossing (90%),
and roundabout (86%). However,
the network was uncertain about the stop sign. It predicted 68% odds that this
was a yield sign and only 10% odds that it was a stop sign. It even offered 
"speed limit 30 km/h" as a third guess with odds of 9%. These sign types are
not similar at all to my eye, and I am confused as to how a network that works
so well on average could make this mistake on what looks like an easy image
to classify.

### Installation


1. Clone the repository

```sh
git clone https://github.com/ericlavigne/CarND-Traffic-Sign-Classifier-Project
```

2. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset of 32x32 traffic sign images, split into training, validation and test sets. Unzip dataset into project directory.

3. Setup virtualenv.

```sh
cd CarND-Traffic-Sign-Classifier-Project
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
deactivate
```

### Running the project

```sh
cd CarND-Traffic-Sign-Classifier-Project
source env/bin/activate
jupyter notebook Traffic_Sign_Classifier.ipynb
deactivate
```

### Installing new library

```sh
cd CarND-Traffic-Sign-Classifier-Project
source env/bin/activate
pip freeze > requirements.txt
deactivate
```
