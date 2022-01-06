# Chiller_Analysis

## Introduction
We are fortunately to have some data of chillers in a commercial building.
To optimize the chiller system operation, we predict the total cooling load over weather and building variables.
Then, we intend to advance the chiller plant optimization.


## Data
The datasets include chiller and weather data and are displayed below.
![Data_table](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/Data_table.PNG)


## Cooling Load Analysis
First, we plot the timeseries of chillers' cooling load.
![RT_total_timeserise](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RT_total_timeserise.PNG)
![Chillers_RT_timeserise](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/Chillers_RT_timeserise.PNG)

Also, the weekly profile of chillers' cooling load is investigated.
![RT_weekly_profile](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RT_weekly_profile.PNG)
![RT_vs_Ta](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RT_vs_Ta.PNG)
![RTmax_vs_Ta](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RTmax_vs_Ta.PNG)

## Cooling Load Prediction
We utilize lgbm for training and predicting.
The details of the training code are in [/src](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/src/Cooling_load_predict.py).
The predictions and the outcomes are plotted below.



Now, we can tell the total cooling load of the building with the weather condition.
Then, we may be able to do the chiller plant optimization while we determine the chillers' total cooling load.

## Chiller Plant Optimization
The content is still processing and will share to you soon.


## Author
- [@Jacky Weng](https://github.com/JackyWeng526)


## Acknowledgement
The module and the application here are just one of the sample works, not the real one in the field.
