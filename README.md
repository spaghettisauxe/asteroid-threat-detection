# Asteroid Threat Detection

A machine learning project that classifies near-Earth asteroids 
as potentially hazardous (impact) or safe (halo) using NASA close 
approach data from 2015 to 2035.

## Results
- Model: Logistic Regression
- Impact Recall: 100%
- Overall Accuracy: 96%

## Features Used
- Distance in lunar units
- Velocity in km/s  
- Absolute magnitude

## Methodology
Target column engineered using NASA's official PHO criteria:
distance <= 19.5 lunar units AND absolute magnitude <= 22.0

## Overview

The model learns patterns from distance, velocity, and size-related features to classify asteroids.
Data is split into training and testing sets (80/20), and performance is evaluated on unseen data.

## Notes

This is a classification problem, not a real-time detection system
The target is derived from known rules, so results reflect how well the model learns those patterns
Useful as a learning project for feature engineering and classification workflows

## Notebook
Full analysis available on Kaggle: [Project JAX](https://www.kaggle.com/code/spaghettisauxe/project-jax)
