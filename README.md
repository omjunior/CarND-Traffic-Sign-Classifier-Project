# Project: Build a Traffic Sign Recognition Program


## Overview

The goal of this project is to design and train a model that can be used to classify German traffic signs.
It was trained using a dataset containing 51,839 classified examples, each color image with 32x32 pixels, divided in the following way:
- 34,799 examples on the training set
- 4,410 examples on the validation set
- 12,630 examples on the test set

The project was implemented on a [Jupyter notebook](Traffic_Sign_Classifier.ipynb). The execution output of this notebook can be seen on [this html](Traffic_Sign_Classifier.html) page.
A visualization of a dataset sample can be found on the notebook.


## Data flow

### Preprocessing

The images were equalized and normalized to ensure that data points are not concentrated over a small excursion.

The equalization method used was transforming the image colorspace to LAB and applying the Contrast Limited Adaptive Histogram Equalization over the lightness channel and then converting it back to RGB colorspace.

Normalization was mean subtraction and standard deviation division.

No further preprocessing was done, like features extraction or applying definite filters, because the model is supposed to derive what it needs on its own.

The training set was augmented by applying some transformations to copies of the images, in order to have more different examples for training the network.
The transformations used were:
- Rotation [-10 deg, +10 deg]
- Scaling [0.9, 1.1]
- Gaussian noise [mean 0, std 0.1]

Preprocessing and transformations were implemented using the OpenCV library.

### Architecture

The network consists of three convolutional layers followed by three fully connected layers. All activation functions were leaky ReLUs. Max pooling was used in every convolutional layer with size and stride of two. Dropout was used on every layer but the last.

Layer | Note | Input | Output | Parameters
--- | --- | :---: | :---: | ---:
Convolutional | 5x5 | 32x32x3 | 28x28x50 | 3,800
Activation | Leaky ReLU |
Max pooling | 2x2 | 28x28x50 | 14x14x50
Dropout | keep 0.67 |
Convolutional | 3x3 | 14x14x50 | 12x12x80 | 36,080
Activation | Leaky ReLU |
Max pooling | 2x2 | 12x12x80 | 6x6x80
Dropout | keep 0.67 |
Convolutional | 3x3 | 6x6x80 | 4x4x100 | 72,100
Activation | Leaky ReLU |
Max pooling | 2x2 | 4x4x100 | 2x2x100
Dropout | keep 0.67 |  
Flattening || 2x2x100 | 400
Fully connected || 400 | 120 | 48,120
Activation | Leaky ReLU |
Dropout | keep 0.67 |
Fully connected || 120 | 80 | 9,680
Activation | Leaky ReLU |
Dropout | keep 0.67 |
Fully connected || 80 | 43 | 3,483
Total | | 32x32x3 | 43 | 173,263


The use of leaky ReLU makes it more likely that all units stay 'alive' during the training process. On a regular ReLU, once it's output is zero it's associated term on the derivative of the cost function disappears, making that unit's input weights not change anymore, making it a 'dead ReLU'.

Max pooling was used to reduce the number of parameters on the network.

Dropout and regularization were used as a form to prevent overfitting.

### Training

The data was split into mini batches of size 128, and run over 20 epochs with a learning rate of 0.0003.
Other hyperparameters used include the alpha for Leaky ReLUs of 0.2, a beta for regularization of 0.01 and drop keep of 0.67.

These values were tuned by training the network over the training set and evaluating accuracy on the validation set.
I believe that these values could be further optimized, but the time consumed for each training session prevented me from reaching it.

I could see that dropout was more effective in preventing overfitting while keeping a good result on the validation set than regularization was.

### Solution

One problem faced was the use of dropout prior to max pooling. The idea of dropping out an output of a layer is to make the next layer independent of a specific input. By using dropout behind max pooling, almost always the max pooling will have a non-zero neighbor value to be used, hence making an input for the next layer available.
Once the order was switched results got better, going from ~92% to ~95% on the validation set.

Overall, once I got results above 95% on the validation set, I run the network on the test set to get 94.17%.

## Testing the model on new data

In order to confirm the model with unseen data, I took some screenshots from Google Maps street view and resized them to 32x32x3.

I have preprocessed the images the same way I did with all the previous data, and the model was able to correctly identify all 5 samples.

The results were pretty firmly certain.
The only example below 90% was a 'no entry' sign, with 88.91% of certainty for the correct label, and 7.37% for a 'stop' sign, which is also red with white markings in the middle.

## Conclusion

Overall the model performed well, but I am certain that with more fine-tuning of the hyperparameters better results could come from the same architecture.
Other than that I have not tried inception modules, which could potentially make the results even better.
But it shows that even a relatively simple model can have a good prediction rate.
