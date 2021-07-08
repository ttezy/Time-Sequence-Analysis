# Time-Series-Analysis

>**Note:**
>
>If you want to learn how each algorithm works, you may check 'VAR_Multivariate.ipynb', 'LSTM_Multivariate.ipynb' and 'Prophet_Multivariate.ipynb' in the main folder.  
>
>If you want to view the performances, you may check '/Performance_Evaluation/Comparison.ipynb'.

# Dataset
'pollution.csv' is the pre-processed data of Beijing PM2.5.

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

