This repository contains code for the model we developed in order to ascertain whether news sentiment can be used to accurately predict stock prices. We implemented and evaluated an LSTM, gradient boosting, and hybrid model, though none of the models we developed were able to strongly predict the test data even if they were well-trained on train data, which we suspect is due to a lack of consideration of several other features of stock data beyond close values. We also made the assumption that stock values would hold over from the previous day if the market was closed, and the sentiment was held over from the previous day if there were no articles from the current day. 

RUNNING THE CODE

In order to run the code, please create a folder called "training_data" in the cloned base of this repository. Then follow the argument parser instructions when running main.py to retrieve data, run the model, and perform a graphing analysis. You also need a premium API key from Alphavantage, whom we source the news sentiment data from. [https://www.alphavantage.co/premium/]

FUTURE ADVANCEMENTS

One way we'd improve this model is to incorporate other features into the gradient boosting model, such as price-to-earnings, PEG, price-to-sales, etc so that it can better pick up on feature correlations in the sentiment. Additionally, we'd change the hybrid model so that the grad boosting just looks at the sentiment stats to find one sentiment value per data, and this is correlated wiht the LSTM.