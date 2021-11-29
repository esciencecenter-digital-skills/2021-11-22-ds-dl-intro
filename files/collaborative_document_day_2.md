![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Deep Learning Day 2

2021-11-23 Introduction Deep Learning

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

## 👮Code of Conduct

* Participants are expected to follow those guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, type `/hand` in the chat window.

To get help, type `/help` in the chat window.

You can ask questions in the document or chat window and helpers will try to help you.

## 🖥 Workshop website

[link](https://esciencecenter-digital-skills.github.io/2021-11-22-ds-dl-intro/)

🛠 Setup

[link](https://esciencecenter-digital-skills.github.io/2021-11-22-ds-dl-intro/#setup)

Download files

[link](https://esciencecenter-digital-skills.github.io/2021-11-22-ds-dl-intro/#downloading-the-required-datasets)

## 👩‍🏫👩‍💻🎓 Instructors

Djura Smits, Sven van der Burg

## 🧑‍🙋 Helpers

Yang Liu, Leon Oostrum

## 🗓️ Agenda
| | |
|-|-|
|09:00|Welcome and icebreaker|
|09:15|Monitoring the training process|
|10:15|Coffee break|
|10:30|Monitoring the training process|
|11:45|Tea break|
|12:00|Monitoring the training process|
|12:45|Recap|
|13:00|END|

## 🔧 Exercises
#### Exercise: Explore the dataset
Let’s get a quick idea of the dataset.

1. How many data points do we have?
2. How many features does the data have (don’t count month and date as a feature)?
3. What are the different measured variable types in the data and how many are there (humidity etc.) ?

Put your answers here
| Name |  |
|------|--|
Artur | 3654 datapoints, 89 features, 81 float64 and 10 int64
Dimitrios | 3654 observations, 89 features, 9 per city
Duong | 
Felicia | 1. 3654, 2. (91 total - Month/Date - 89), 3. Pressure, precipitation, Cloud cover...
Florian | 3654, 89total /9 per city (some cities don't have the same amount?), 
Jelle | 3654 | 89 | 
Khun Sang | (1) 3654, (2) 89, (3) 9
Kwabena Kingsley | 1) 3654; 2)
Lukasz | (1) 3654; (2) 9; (3)
Pesi | 3654, 163, 
Rodrigo | 3654, 89, 
Ronald | 3654, 163, 8
Sam | 3654, 163, not all cities have same variables e.g. Basel and de Bilt
Sobhan | 1. 3654
Thang | (1) 3654 (2) 89
Tugce |1)3654 2)89 3)9

The correct answers are:
We can get the number of features by printing the shape of your data by 
```python=
data.shape
```
This gives 3654 x 89, which are the data points we have.

In order to print the features the data have, we do
```python=
{x.split('_'[-1] for x in data.columns if x not in ['MONTH', 'DATE'])}
```

#### Exercise: Split data into training, validation and test set
Split the data into 3 completely separate set to be used for training, validation, and testing using the train_test_split function from sklearn.model_selection. This can be done in two steps. First, split the data into training set (70% of all datapoints) and validation+test set (30%). Then split the second one again into two sets (both roughly equal in size).

- How many data points do you have in the training, validation, and test sets?

Put your answers here

| Name |  |
|------|--|
Artur | Training = 767, testing = 164, validation = 165
Cristian |
Dimitrios |train 767, test 165, val 164
Duong | 767,329
Felicia | Train: 767,  329 (splitted into 165, 164)
Florian |767, 165/164
Jelle | 767, 164, 165
Khun Sang | training: 767, testing: 164, validation: 165
Kwabena Kingsley | training 767, validation 164, testing 165
Lukasz | Training: 767; Testing: 164; Validation: 165
Pesi | 767, 165, 164
Rodrigo
Ronald | train: 767, validate: 164, test: 165
Sam | Train: 767, Test:164, Val: 165
Sobhan |
Thang | Train (767); Val (164); Test (165)
Tugce |767,164,165
Leonoor |  X_val: 231, X_test: 329, X_train: 536

Hint: think about dividing your data like
```python=
X_train, X_not_train, y_train, y_not_train = train_test_split(X, y, test_size=0.3, random_state=42)
```
Caveat: do we need shuffling when preparing our data?

The correct answers are:
training set: 767
validation set: 164
test set:165

#### Exercise: Create the neural network
We have seen how to build a dense neural network in episode 2. Try now to construct a dense neural network with 3 layers for a regression task. Start with a network of a dense layer with 100 nodes, followed by one with 50 nodes and finally an output layer. 

Hint: Layers in keras are stacked by passing a layer to the next one like this.
```python=
inputs = keras.layers.Input(shape=...)
next_layer = keras.layers.Dense(..., activation='relu')(inputs)
next_layer = keras.layers.Dense(..., activation='relu')(next_layer)
#here we used the same layer name twice, but that is up to you
...
next_layer = ...(next_layer)
...
#stack as many layers as you like
```

- What must here be the dimension of our input layer?
- How would our output layer look like? What about the activation function? Tip: Remember that the activation function in our previous classification network scaled the outputs between 0 and 1.


Put your answers here:
| Room |  |
|------|--|
1 |89, output 1 node, Relu activation
2 |163 (or 89 for light dataset), Should last layer (output) have Relu activation? Cant have less than 0 sunshine hours (But also not more than 24)
3 |
4 | 88 (89 minus 1 BASEL_sunshine), output layer: ```keras.layers.Dense(1, activation="relu")```
5 | 1) 89; 2) ```keras.layers.Dense(1, name="Output")(hidden_2_layer)```

#### Exercise: Predict the labels for both training and test set and compare to the true values
Even though we here use a different model architecture and a different task compared to episode 2, the prediction step is mostly identical. Use the model to predict the labels for the training set and the test set and then compare them in a scatter plot to the true labels.

Hint: use `plt.scatter()`. 
Hint: the predicted labels can be generated using y_predicted = model.predict(X).

- Is the performance of the model as you expected (or better/worse)?
- Is there a noteable difference between training set and test set? And if so, any idea why?
 
Hint: you can use model.evaluate to obtain metric scores for train and test set.

Put your answers here:
| Name |  |
|------|--|
Artur | The model performed worse than I expected, based on the MSE reported. Comparing the training and test sets, the model has clearly overfitted.
Cristian |
Dimitrios | 
Duong | 
Felicia |
Florian |
Jelle | performance is not great. I think this has to do with overfitting.
Khun Sang | model did not perform well on test data
Kwabena Kingsley |
Leonoor |
Lukasz | Model performed well on training set with low loss/mse (MSE: 0.6) but not so well on the test data (MSE: 4.7). Clearly overfit.
Pesi | worse in loss training
Ronald | test-set is bad, train-set is good
Sam | Test quite a bit worse than training.
Sobhan |
Thang | loss train = 0.6, loss val = 22.3, loss test = 16.8. Quite a few negative values in predictions.
Tugce |The model performed poorly with test set based on the graph. 

#### Exercise: Create a baseline and plot it against the true labels
Create the same type of scatter plot as before, but now comparing the sunshine hours in Basel today vs. the sunshine hours in Basel tomorrow. Also calculate the RMSE for the baseline prediction.

Hint: you can use: 
```python=
from sklearn.metrics import mean_squared_error
rmse_score = mean_squared_error(true_values, predicted_values, squared=False)
```

- Looking at this baseline: Would you consider this a simple or a hard problem to solve?

Put your answers here:
| Room |  |
|------|--|
1 |Based on the plot, hard problem as there's no obvious patterns. Based on comparing the plots, the NN model is slightly better, but the RMSE score is similar (both around 4.3).
2 | Naive MSE: 25.7 RMSE:5.07, model MSE: 20.1 RMSE: 4.48
3 |the RMSE (approx. 4.2) of the toy model is worse than our dl model (approx. 3.6)
4 |
5 | We would have to judge based on the performance of our model, however, based on the scatter plot it would not be straighforward.


#### Exercise: Add a BatchNormalization layer as the first layer to your neural network.
Look at the [documentation of the batch normalization layer](https://keras.io/api/layers/normalization_layers/batch_normalization/). Add this as a first layer to the model we defined above. Then, train the model and compare the performance to the model without batch normalization.


#### Open question: What could be next steps to further improve the model? (breakout rooms)
With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results. Usually models are “well behaving” in the sense that small chances to the architectures also only result in small changes of the performance (if any). It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist? Applying common sense is often a good first step to make a guess of how much better could results be. In the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision. But how much better our model could be exactly, often remains difficult to answer.

- What changes to the model architecture might make sense to explore?
- Ignoring changes to the model architecture, what might notably improve the prediction quality?

## 🧠 Collaborative Notes
### Monitor the training process
1. Define the problem and check the dataset
We use pandas to manage the data
```python=
import pandas as pd
```
Link to the dataset (weather data)

```python=
data = pd.read_csv("https://zenodo.org/record/5071376/files/weather_prediction_dataset_light.csv?download=1")
```
Check the data
```python=
data.head()
```
Print the list of variable names in the column
```python=
data.columns
```

Prepare dataset for the inspection of sunshine hours.

Exercise: line 114

We first check the first three years of data.

```python=
nr_rows = 365 * 3
X_data = data.loc[:nr_rows].drop(columns=['DATE', 'MONTH'])

y_data = data.loc[1:(nr_rows + 1)]['BASEL_sunshine']
```
Note that for `y_data` we start from point one because we want to make prediction from today to tomorrow.

To have an overview of our data
```python=
data.describe()
```

We will split the data. In deep learning, except for training data nad testing data, it is important to have a separate validation set.

But before spliting the dataset, let's first choose the random state for today (we pick number 42).
```python=
from random import seed
seed(42)
```

Setup the random seed for `tensorflow`
```python=
from tensorflow.random import set_seed
set_seed(42)
```

And for `numpy`
```python=
from numpy.random import seed
seed(42)
```

Exercise: line 153

2. Split data into training set, validation set and testing set

We use the data hanlding module in `sklearn` to split our data
```python=
from sklearn.model_selection import train_test_split
```

Further split the "not training set" to validation and testing sets
```python=
X_val, X_test, y_val, y_test = train_test_split(X_not_train,
                                                y_not_train, 
                                                test_size=0.5, 
                                                random_state=42)
```

Let's see how much data do we have for training, validation and testing sets:
```python=
print(f'Data was split into training ({X_train.shape[0]}),'\
      f'validation ({X_val.shape[0]}) and test set ({X_test.shape[0]}).')
```

3. Create neural network

Now let's build our neural network. Based on the task for today, which is regression now, we will have different configurations for our network.

For instance, this has an impact on our choices of loss function. Yesterday we performed classification with our network and we used categorical cross-entropy. Now we are doing regression, we need something different.

Exercise: line 192

First, we create our neural network using keras. We will construct a dense neural network with 3 layers for a regression task. Start with a network of a dense layer with 100 nodes, followed by one with 50 nodes and finally an output layer. 
```python=
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization

def create_nn(first_nodes=100, second_nodes=50):
    inputs = keras.Input(share=(X_data.shape[1]), name='input')
    
    # layer to perform batch normalization
    layers_dense = keras.layers.BatchNormalization()(inputs)
    
    layers_dense = keras.layers.Dense(first_nodes, 'relu')(layer_dense)
    layers_dense = keras.layers.Dense(second_nodes, 'relu')(layers_dense)
    
    outputs = keras.layers.Dense(1)(layers_dense)
    
    return keras.Model(inputs =inputs, 
                       outputs=outputs, 
                       name='weather_prediction_model')
    
model = create_nn()
model.summary()
```

Before compiling the model, we need to define a few things, like loss function, and optimizer. 

We need to choose a loss function. Here are several options for a regression task, for example:
- mean square error (mse)
- mae (mean absolute error)

MSE is often preferable to MAE, because it "punishes" predictions that deviate further from the truth more.

And for optimizer, there are also a few options, for example:
- Adam optimizer
- stochastic gradient descent

Now we can compile our model

```python=
model.compile(optimizer='adam', 
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])
```

4. Train the neural network

Before start training, we need to understand another concept, which is 'batch size'. Batch size refers to the number of training examples utilized in one iteration. It has influence on the optimization based on the gradients.

Now let's start the training process
```python=
history = model.fit(X_train, y_train,
                    batch_size=32,
                   epochs=200,
                   verbose=2)
```

Our training history looks like
```python=
history.history.keys()
```

We can visualize our training history by making some plots.
```python=
import seaborn as sns
import matplotlib.pyplot as plt

history_df = pd.DataFrame.from_dict(history.history)
history_df.head()
```

5. Evaluate the training and make predictions

We have put the training history into the pandas pipeline. We can create some lines plots based on it.

```python=
sns.lineplot(data=history_df['root_mean_squared_error'])
plt.xlabel('epochs')
plt.ylabel('RMSE')
```
Exercise: line 220

Now we can predict the labels for both training and test set and compare them to the true values:
```python=
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)
```

Let's plot these predicted results
```python=
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(y_train_predicted, y_train, alpha=0.5, s=10)
axes[0].set_title('training set')
axes[0].set_xlabel('predicted sunshine hours')
axes[0].set_ylabel('true sunshine hours')

axes[1].scatter(y_test_predicted, y_test, alpha=0.5, s=10)
axes[1].set_title('training set')
axes[1].set_xlabel('predicted sunshine hours')
axes[1].set_ylabel('true sunshine hours')
```

We can inspect the loss by evaluating the model
```python=
loss_train, rmse_train = model.evaluate(X_train, y_train)
loss_train, rmse_train = model.evaluate(X_test, y_test)

print(f'Train RMSE: {rmse_train:.2f}, Test RMSE: {rmse_test:.2f}')
```

The test resuls are not very good, compared to the trainging results. We find the model is clearly overfitted in this case. We need to consider how to avoid overfitting.

Exercise: line 261

6. Evaluate the predictions

We need to setup a baseline prediction for the purpose of diagnostics
```python=
from sklearn.metrics import mean_squared_error

y_baseline_prediction = X_test['BASEL_sunshine']

plt.figure(figsize=(5,5), dpi=100)
plt.scatter(y_baseline_prediction, y_test, s=10, alpha=0.5)
plt.xlabel('sunshine hours yesterday')
plt.ylabel('true sunshine hours')
```
And we can better evaluate our training by comparing the error from our prediction and the baseline:
```python=
from sklearn.metrics import mean_squared_error

rmse_nn = mean_squared_error(y_test, y_test_prediction, squared=False)
rmse_baseline = mean_squared_error(y_test, y_baseline_prediction, squared=False)

print(f'NN RMSE: {rmse_nn:.2f}, baseline RMSE: {rmse_baseline:.2f}')
```

We create a new network and include the validation data during training
```python=
model = create_nn()
model.compile(optimizer='adam', 
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    verbose=2)
```

We can also plot the training process
```python=
history_df = pd.DataFrame.from_dict(history.history)

sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
plt.xlabel('epochs')
plt.ylabel('RMSE')
```

It is always necessary to decide when to stop the training. We can use "early stopping" for this.

To implement the early stopping feature, we create another network

```python=

verbose=1)

history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=200,
                    validation_data=(X_val, y_val),
                    callbacks=[earlystopper]
                    verbose=2)
```

And we check the training process again by plotting them
```python=
history_df = pd.DataFrame.from_dict(history.history)

sns.lineplot(data=history_df[['root_mean_squared_error', 'val_root_mean_squared_error']])
plt.xlabel('epochs')
plt.ylabel('RMSE')
```

To improve the performance of our training, one strategy is to nomalize our data. This can be done during the training process by adding a batch normalization layer (see line 384)

Exercise: line 274

After implementing the batch normalization during training, let's see how are our predictions now

```python=
y_test_predicted = model.predict(X_test)

plt.figure(figsize=(5,5), dpi=100)
plt.scatter(y_test_predicted, y_test, s=10, alpha=0.5)
plt.scatter(y_baseline_prediction, y_test, s=10, alpha=0.5)
plt.xlabel('predicted sunshine hours')
plt.ylabel('true sunshine hours')
```
Exercise: line 278 (let's take some time to think about it :coffee:)


### Tips:
- Time is not very convenient. I have a lot of  meetings in the morning, not sure about others. 
- I would like to see some advice regarding "best practices" incorporated in the material.
- Occasionally you scroll down a bit too quickly when writing code, especially when a big picture like a graph appears. Makes it difficult to copy the commands needed for the API.
- Once I diverge from the code by the instructor (variable names, functions, etc.) I get in problems a a later moment. Maybe you should press the need to follow exactly the code.
- I would like to hear about other methods to avoid overfitting, underfitting :+1: :+1:
- Djura, drink more water! :smile: or alcohol... :smiley: 
- It might help if you could repeat the Batch Normalisation methods in DL or one exercise on Batched Normalisation.
- might help to have some additional 'mini-exercises' inbetween, e.g. questions about the background of using a specific metric, variable etc. 
- It would be nice to have a guide of improving the model like a flowchart. If X do Y, etc
- I got a bit lost in the beginning, trying to understand the case study. Maybe 1-2 slides explaining the dataset and what we are going to do would make things easier!

### Tops:
- The excercise is flexible for people from different background. :+1:
- It is nice to meet people in the break out room as well. :+1:
- I enjoyed today's sessions. Following the instructor is a a fun and relaxing way of learning.:+1:
- I love the shorter course days, they make the course much more accessible for people with disabilities like myself. Also leaves some room to still do mandatory work if needed. :+1:
- Great pace, and very interesting again today! :+1: 
- The explanation of the data was a bit confusing, regarding yesterday and today sunshine values.:+1:
- The concept (overall structuring of the topics) is easy to follow. Even if you need to look up something inbetween (might happen for me) :+1:
- Very interesting today. Learn quite a bit about the Keras' API. :+1: :+1:

## :car: Parking Lot

## 📚 Resources
- 'Practical deep learning for coders'[Fast.ai](https://course.fast.ai/). Claims to 'make neural nets uncool again'.
- [Keras code examples](https://keras.io/examples/)
- [deep learning course offered by DeepLearning.AI](https://www.coursera.org/specializations/deep-learning?)
- ['Machine learning yearning' by Andrew Ng](https://github.com/ajaymache/machine-learning-yearning). Small one-page practical tips, about real-world machine learning setting that you often don't learn about in courses. For example: 'How to decide how to include inconsistent data?'.