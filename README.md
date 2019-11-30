# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, a driving behavior is cloned using deep neural networks and convolutional neural networks. The model architecture is trained, validated and tested using Keras. The model outputs a steering angle to an autonomous vehicle, run on a Udacity provided simulator.

The simulator provided can steer a car around a track for data collection. Using image data and steering angles to train a neural network,  the model is then used to drive the car autonomously around a track.

The project has five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* AutonomousMode_Track1.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)


---
The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior.
* Design, train and validate a model that predicts a steering angle from image data.
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report.