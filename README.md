## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting three files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report either as a markdown or pdf file 

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/481/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 
The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

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
