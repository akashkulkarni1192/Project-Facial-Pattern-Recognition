# Project-Facial-Pattern-Recognition
Improved ANN model's result by adding CNN layers on top of it

#### Goal ####
Given an image of a face, categorize it based on the emotion shown on its facial expression

#### Dataset ####
Kaggle's facial expression challenge's dataset has been used for this project. The training dataset contains 28,709 samples of 48x48 grey scale images of faces.
Dataset can be found on this link : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

#### Emotion can be of following types ####
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

#### Implementation ####
* **ANN with momentum** : I first implemented using ANN model with 2 hidden layers with 500 and 300 activation units respectively, and regularization. But since the learning was very slow, I used momentum techinque to improve the learning time. With this model, I was able to get the **best error rate of 0.17**
* **CNN with momentum** : I then used two convolution-pooling layers over ANN model. Each convolution filter was of size height=5 width=5 and feature map outs as 50. With this model, I was able to get the **best error rate of 0.02**
