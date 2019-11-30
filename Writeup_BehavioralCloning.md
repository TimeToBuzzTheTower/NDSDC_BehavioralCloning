# **Behavioral Cloning** 

## Project Summary

### The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./WriteUpImages/WriteUp_SteeringDistribution.png "Steering Angles Distributions"
[image3]: ./WriteUpImages/WriteUp_NVIDIA_CnnArchitecture.png "NVIDIA CNN Architecture"
[image4]: ./WriteUpImages/WriteUp_AutonomousDrivingImage.PNG "Autonomous Driving Track 1"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### The project submission includes the following files which are required to run the simulator in autonomous mode: 

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 

#### 1. Data Collection for Training the neural network
Using the Udacity provided simulator, 

We want to avoid overfitting or underfitting when training the model. Hence it is essential to know

* when to stop collecting data
* ensuring a clean driving procedure and repeated accumulation of data over the sections, ensuring a consistent pattern as much as possible.
* Increasing variants by actually driving in the other direction to get further perspective of the road topologies and the related steering wheel angles.
* flipping the images is a quick way to augment the data.
* collecting data from the second track can also help generalize the model.

The accumulated data was used for training the convolutional network. In addition, for initial training purposes and ensuring a proper execution, the dataset provided by Udacity was used.

![alt text][image2]

***Ilustration of steering angle distributions of collected data of Udacity dataset***

#### 2. Model Architecture and Training Strategy
Taking upon the suggestion of the Udacity tutorials and various debugging attempts with the LeNet Network (in addtion to some useful exchanges and guidance from Udacity mentors) the nVidia Autonomous Car Network Architecture, offering the minimization approach through mean-squared error between the steering command output by the network, the recorded steering commands through the simulator in the csv file.

![alt text][image3]

***NVIDIA CNN Architecture with Convolutional Layers and fully connected layers***

The layer for single output is attached at the end as required. The primary principle adopted here is based on:
* Small size of dataset and 
* Greater similarity of training data, considering the data is gathered over a vehicle simulator, but on a road image.

Not performing flawlessly in the autonomous mode at certain locations, it was evident that a correction could probably be required for the side camera images (left and right) to append the steering measurements further with minimal error. Augmenting the data by adding the same image flipped and indtroducing a 0.2 correction, likewise to the left and right images essentially helped the car to try and keep to the center of the lane.
The data was further augmented by acquiring data from the other complicated track too, which had gradients and approaches needed to not only keep a good driving line, but also to ensure correct braking procedures.

##### Command used for creating the model #####
```sh
python model.py model.h5
```
##### Statistics for training and validating the model
Total Images: 24108
Training samples: 19286
Validation samples: 4822

Epoch 1:
19286/19286 [==============================] - 2467s 128ms/step - loss: 0.0113 - val_loss: 0.0187

Epoch 2:
19286/19286 [==============================] - 2461s 128ms/step - loss: 0.0078 - val_loss: 0.0217

Epoch 3:
19286/19286 [==============================] - 2454s 127ms/step - loss: 0.0071 - val_loss: 0.0217

Layer (type)                 Output Shape              Param #   

lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 45, 160, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 23, 80, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 40, 48)        43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 10, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 1920)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               192100    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        

Total params: 329,019

Trainable params: 329,019

Non-trainable params: 0
_________________________________________________________________

#### 3. Final Result: Track 1

Following is an illustration of the simulator vehicle driving autonomously on Track 1. The video as a result file (avi) is uploaded on the Github repository.

<a href="https://youtu.be/BYOy_TRDy3c" target="_blank"><img src="https://i9.ytimg.com/vi/BYOy_TRDy3c/mq2.jpg?sqp=CJWwiu8F&rs=AOn4CLAudyRSZP7uzuBo_BM3bTgsVtkA1A" 
alt="Final Result: Track 1" width="460" height="220" border="1" /></a>

---
## Conclusions and lessons learned

The project highlights the following aspects towards the field of Deep learning, namely:
* The extend of CNNs and their advancements at various companies and industries
* The maturity of the system and the upcoming usages and usecases where it could be applied (banking, investments, environment predictions, to name a few...)
* Specific to autonomous vehicles: Initiating and validating the debate about: How much and which sensors do we actually need for the purpose, which could be easily served by CNNs.

As a seasoned OEM Engineer, having been through many serial development vehicle projects, it always boils down to the cost for the idea or service offered. Hence, providing functionalities are always gauged against the extend of costs, the project is willing to invest.
Sensors and their processing power requirements are certainly cost drivers, which could have been limiting the possibilities of employing self driving cars (more conservatively speaking: Advanced Driver assistance systems and Pilots). In addition, package, cooling and durability bears to dawn the realities that engineers face towards making a system happen.
CNNs and Deep Learning definitely supports the systems engineers to the extend of providing an alternative towards offering a functionality, which could be in the realm of functional safety, a valid, reliable and economical option - rather than stocking up vehicles with sensory and camera systems, which predominantly have a voluminous of redudant operations and resource consumptions (electrical energy from HV Batteries, increased fuel consumption, cooling systems requiring substantial on-board energy).

The field of Deep Learning is a complete new expertise by itself. It would take some generations for a paradigm shift in employing the methods and maturities in traditional vehicle companies, which are seasoned and homologated on conventional methodologies and still bank of the responsibility of the one essential controller of the whole system: The HUMAN DRIVER.

Further improvements for the project would essentially be:
* Improved data collection for a better modelling.
* reducing the number of EPOCHS and exploiting other Neural Networks for a qualitative comparison.