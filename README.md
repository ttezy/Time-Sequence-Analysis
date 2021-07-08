
# Time-Series-Analysis

## Files

#### Main Folder
'pollution.csv' - dataset

VAR_Multivariate.ipynb - Implementation of VAR Model on 'pollution' dataset
LSTM_Multivariate.ipynb - Implementation of LSTM Model on 'pollution' dataset
LSTM_Multivariate_traffic.ipynb - Implementation of LSTM Model on 'trafffic' dataset
FB_Prophet_Multivariate.ipynb - Implementation of FB Prophet on 'pollution' dataset

#### Performance_Evaluation - folder that contain performances of different algorithms
VAR_Multivariate_Performance.ipynb - Performance of VAR Model
LSTM_Multivariate_Performance.ipynb - Performance of LSTM Model
FB_Prophet_Multivariate_performance.ipynb - Performance of Prophet Model

>**Note:**
>You can learn how algorithms work by viewing the files in the main folder
>You can check different performances by viewing the files in the Performance_Evaluation folder

  

# Dataset

The raw dataset is like this:
|No| year | month | day | hour | pm2.5 | DEWP | TEMP | PRESS | cbwd | Iws | Is | Ir
|--|--|--|--|--|--|--|--|--|--|--|--|--|
|1|2010|1|1|0|NA|-21|-11|1021|NW|1.79|0|0|
|2|2010|1|1|1|NA|-21|-12|1020|NW|4.92|0|0|
| ... |...|...|...|...|...|...|...|...|...|...|...|...|

Processed data is like this:

|date| pollution | dew | temp | press | wnd_dir | wnd_spd | snow | rain
|--|--|--|--|--|--|--|--|--|
| 2010-01-01 00:00:00 |0.0|-21|-11.0|1021.0|NW|1.79|0|0
| 2010-01-01 01:00:00 |0.0|-21|-11.0|1020.0|NW|4.92|0|0
| 2010-01-01 02:00:00 |0.0|-21|-11.0|1019.0|NW|6.71|0|0
| ... |...|...|...|...|...|...|...|...
| 2014-12-31 19:00:00 |8.0|-23|-2.0|1034.0|NW|231.97|0|0
| ... |...|...|...|...|...|...|...|...

- 'pollution.csv' is the pre-processed data of Beijing PM2.5, with renamed columns. All 'NA' values are marked as 0.
- The column 'pollution' is the y value that we will use different algorithms to predict. 
- 'date' is the index with one hour for each step.
- Other columns are variables that influence and decide the value of the pollution, which should be considered in the training and prediction procedure.

 **Note:**
 To prepare for the training and testing, additional work need to be done
 - The attribute 'wnd_dir' is string. Models only accept numerical datas, so 'wnd_dir' will be encoded which originally is in type of string.
>
    values = df.values
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
- In this project, 90% data is used for training and the remaining 10% data is used for testing.

# Multivariate Time Series Forecast Algorithms:

  

## Traditional methods:

  

### 1. VAR - Vector Autoregression

  

Tutorial:

[Multivariate Time series using Vector Autoregression (VAR)](https://www.youtube.com/watch?v=TpQtD7ONfxQ)

  

![Image of VAR model](https://slidetodoc.com/presentation_image/5cf1d107218d387627d94c115edfd4bf/image-2.jpg)

  
  

## Machine Learning methods:

  

### 1. LSTM - Long Short-term Memory

  

Tutorial:

[Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)

  

![Image of LSTM model](https://www.researchgate.net/profile/Xuan_Hien_Le2/publication/334268507/figure/fig8/AS:788364231987201@1564972088814/The-structure-of-the-Long-Short-Term-Memory-LSTM-neural-network-Reproduced-from-Yan.png)

![Image of LSTM model](https://www.researchgate.net/profile/Savvas-Varsamopoulos/publication/329362532/figure/fig5/AS:699592479870977@1543807253596/Structure-of-the-LSTM-cell-and-equations-that-describe-the-gates-of-an-LSTM-cell.jpg)

![IImage of LSTM model](https://miro.medium.com/max/1400/1*0R9LrwwY4zd585qEAgws6w.png)

  
  

### 2. Facebook Prophet

  

Tutorial:

[Time Series Forecasting with Facebook Prophet and Python in 20 Minutes](https://www.youtube.com/watch?v=KvLG1uTC-KU)

[Multivariate Time Series Modeling using Facebook Prophet](https://www.youtube.com/watch?v=XZhPO043lqU)

  

For FB Prophet, if you want to forecast more than one future step, you may need to predict all variables first, as one output 'yhat' is corresponding to each row of varables. Thus, if you don't have future value of your variiables, you cannot predict more steps.
