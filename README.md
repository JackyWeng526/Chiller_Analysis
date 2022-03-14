# Chiller_Analysis

## Introduction
We are fortunate to have some data on chillers in a commercial building.

To optimize the chiller system operation, we predict the total cooling load over weather and building variables.

Then, we intend to advance the chiller plant optimization in the near future.

**note: The repository is not complete yet and will be updated irregularly.

## Data
The dataset include chiller and weather data and are displayed below.

The dataset is displayed as long-data classified by chiller name (subset_column==Chiller).

![Data_table](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/Data_table.PNG)


## Cooling Load Analysis
First, we plot the time series of the chillers' cooling load.

![RT_total_timeserise](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RT_total_timeserise.PNG)

![Chillers_RT_timeserise](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/Chillers_RT_timeserise.PNG)

Second, we can look into the energy and cooling performance of the chillers and chiller plants (including energy use of pumps and cooling tower).

![KWRT_total_timeserise](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/Chillers_performance_timeserise.PNG)

![sysKWRT_timeserise](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/System_performance_timeserise.PNG)

Besides, the weekly profiles of the chillers' cooling load are investigated.

![RT_weekly_profile](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RT_weekly_profile.PNG)

![RT_vs_wetT](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RT_vs_wetT.PNG)

![RTmax_vs_wetT](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RTmax_vs_wetT.PNG)

**note: Here used outdoor wet bulb temperature because we found that cooling load patterns were highly relavent to both dry bulb temperature and relative humidity.

## Cooling Load Prediction
We utilize the LightGBM model for training and predicting.

The details of the training code are in the [/src](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/src/Cooling_load_predict.py) folder.

The outcomes are demonstrated below.

![RT_predictions](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RT_predictions.PNG)

![RT_predictions_daymax](https://github.com/JackyWeng526/Chiller_Analysis/blob/main/docs/RT_predictions_dailymax.PNG)

Now, we can tell the total cooling load of the building with the given weather condition (the result of hourly predictions are saved as a csv file).

Then, we may be able to do the chiller plant optimization while we determine the chillers' total cooling load.

## Chiller Plant Optimization
The content is still processing and will share to you soon.


## Author
- [@Jacky Weng](https://github.com/JackyWeng526)


## Acknowledgement
The module and the application here are just one of the sample works, not the real one in the field.
