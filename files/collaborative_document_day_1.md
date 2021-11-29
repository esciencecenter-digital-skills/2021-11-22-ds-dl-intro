![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Deep Learning Day 1

2021-11-22 Introduction Deep Learning

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
|09:30|Introduction|
|10:15|Coffee break|
|10:30|Classification by a Neural Network using Keras|
|11:45|Tea break|
|12:00|Classification by a Neural Network using Keras|
|12:45|Recap|
|13:00|END|

## 🔧 Exercises

### Introduction
#### Deep learning workflow exercise
Think about a problem you’d like to use Deep Learning to solve. As a group, pick 1 problem.

1. What do you want a Deep Learning system to be able to tell you?
2. What data inputs and outputs will you have?
3. Do you think you’ll need to train the network or will a pre-trained network be suitable?
4. What data do you have to train with? What preparation will your data need? Consider both the data you are going to predict/classify from and the data you’ll use to train the network.

Discuss your answers in your breakout group. Write them (concisely) down in the collaborative document.

- Room 1: 
    1. Image classification: Health status / nutrition / phenotyping of a plant 
    2. Input: Example images, Output: class (status, string). 
    3. Pre-trained if trained on existing dataset designed for a rather similar problem. 
    4. Dataset: Images for classes (phenotypes, status); Train/Classify/Evaluate datasets. Labeled train-dataset, pre-processing: Crop, Clean the background, Scale, resolution(?) - for feature detection....
- Room 2: Image calssification: 
    1) if the image contains a cat or dog; data input
    2) a lot of pictures with cats and dogs and settings without cats and dogs 
    3) pretrained network will be suitable
    4) image scalling to have  the same resolution as input   
- Room 3:
    1. Phosphosite - kinase relationship
    2. We have about 10000 known relationship
    3. Ideally we would be able to exploit the large set of protein-protein interaction datasets to improve our protein-kinase interaction.
    4. Mass spectrometry data with phosphorylation capture.
- Room 4:
    1. Recognize music types (styles)
    2. Input: Audio fragments / Output: Type of music
    3. Which audio is which style; yes it would work with pre-trained network (if this exists)
    4. Audio, Frequency spectrum, Loudness, Title (?), Year of release (?)
- Room 5:
    1. Predicting structural DNA element and their regulating patterns (genes, promoters, enhancers etc)
    2. Input: Experimentally validated interactions & annotated structural elements.
    3. Train the network 
    4. Experimentally validated data, Train/test splits. 
- Room 6:
    1. Classification of objects in different biological structures
    2. Inputs/outputs
        1. Inputs: mass spectrometry data of sample(s) (vector of intensities for particular mass-to-charge ratios) and ion images for specific m/z ratio
        2. Output: Vector with the class of each pixels
    4. There might not be a network that's pre-trained but we should be able to reuse existing architecture
    5. Pre-processing and preparation
        1. Fair amount of MS data with pre-existing segmentation masks.
        2. Data needs to be pre-processed because there is some variation between samples (e.g. patients)



#### Check your Python installation
```python=
from tensorflow import keras
print(keras.__version__)

import seaborn as sns
print(sns.__version__)

import sklearn
print(sklearn.__version__)
```

### Classification by a Neural Network using Keras

#### Have you worked with pandas before?
Artur :thumbsup: 
Azin 
Cristian
Diah :thumbsup:
Dimitrios :thumbsup: 
Duong
Felicia :thumbsup: 
Florian :thumbsup: 
Iris
Jelle :+1: 
Khun Sang :+1:
Kwabena Kingsley :+1: 
Leonoor
Lukasz :+1: 
Mario: :thumbsup: 
Pesi
Rodrigo      
Ronald :+1: 
Sam - :thumbsup: 
Sobhan :thumbsup:
Thang :+1: 
Tugce:thumbsup: 
Xiao
Yakob
Shengnan :+1: 

#### Exercise: Inspect the penguins dataset
Inspect the penguins dataset.

1. What are the different features called in the dataframe?
2. Are the target classes of the dataset stored as numbers or strings?
3. How many samples does this dataset have?

You can see the column names (= features) with one of the following commands:
```python=
penguins.head()
penguins.columns
```
The features are species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex

The target classes is the species column:
```python=
penguins['species'].unique()
```
```python=
penguins.dtypes
```
There are 3 species and they are stored as strings.


Finally, the number of samples can be retrieved with
```python=
penguins['species'].describe()
```
There are 344 entries.

#### Exercise: Create the neural network
With the code snippets above, we defined a keras model with 1 hidden layer with 10 neurons and an output layer with 3 neurons.

1. How many parameters does the resulting model have?
2. What happens to the number of parameters if we increase or decrease the number of neurons in the hidden layer?


| Name |  |
|------|--|
Artur | 83 parameters, if the hidden layer gains 1 neuron, the number of parameters increases by 10
Azin |
Cristian |
Diah | 1. 83 parameters 2. increase the number of neurons-->increase parameters
Dimitrios | 83 trainable params, increase neurons -> Increase params, Decrease neurons  -> Decrease num of params 
Duong | 
Felicia | 1. 83 parameters 2. weights per neuron, so increase -> increase, decrease -> decrease 
Florian |83 param, increases/decreases based on the number of neurons
Iris | 
Jelle | 83 param 2. increases when number of neurons increases
Khun Sang | 83 params,
Kwabena Kingsley | 1. 83 parameters, 2. increasing the number of neurons increases the parameters and vice versa
Leonoor | 83 parameters (bias, and , more neurons = more parameters
Lukasz | (1) 83 parameters (2) increase -> increase, decrease -> decrease
Mario | (1) 83 (2) increase/decrease :: there is (bias + weight x num_features) per neuron, thus 4 features * 10  + 10 = 50.
Pesi | 83 params, increases /decreases performances of recognition
Rodrigo | 83, increases/decreases
Ronald | 83, the number of parameters is the summation of the parameters between all layers. These numbers depend on the number of nodes in the previous layer **plus one** times the number of nodes in the current layer. *Why the **plus one**? Is this bias?*
Sam | 83 parameters (weights+bias). Increase=More parameters, decreate=Less. 1 bias per neuron + 1 x neurons previous layer weights
Samer |
Shengnan |
Sobhan | 83 params, decreasing the number of neurons decreases the number of params, increasing the number of neurons increases the number of params
Thang | 83 = (4 x 10 + 10) + (10 x 3 + 3), an increasing function with respect to the numbers of neurons, number of layers, inputs, outputs.
Tugce |1- 2- it doesn't change
Xiao |
Yakob |
Xuetong |

#### How do you think the model will perform on the test set?
| Name |  |
|------|--|
Artur | The training set is on the smaller size, but I think the problem is easy enough for the model to do okay.
Azin |
Cristian |
Diah | The performance is not as good as in the training
Dimitrios | Maybe the model is too flexible?
Duong | 
Felicia | should perform well
Florian |Most likely will suffer from overfitting
Iris | 
Jelle | not very well due to overfitting
Khun Sang |
Kwabena Kingsley | 
Leonoor |
Lukasz | based on the loss value, it will perform well but its going to overfit
Mario | what is well> but fairly ok sure for a first model ~similar loss the data set seems fairly straightforward
Pesi | yes
Rodrigo | yes
Ronald | 
Sam | It should perform well on the test set if all data is relatively similar
Shengnan |
Sobhan |
Thang | yes, it will perform well.
Tugce |It'll perform well
Xiao |
Yakob |
Xuetong |?

## What could we do to improve the results?
| Name |  |
|------|--|
Artur | Expand trainingset, smaller test set. More epochs, or a complexer model with more layers.
Azin |
Cristian |
Diah | using data augmentation
Dimitrios | ?
Duong | 
Felicia |no. of iterations (epochs increase), increase size of training-dataset?
Florian |Use less biased trainingdata/a more representative set, more/less nodes, more/less layers (more probably helps more given the current overfitting on one class).
Iris | 
Jelle | reduce number of parameters/neurons
Khun Sang | more training data
Kwabena Kingsley | 
Leonoor |
Lukasz | Add more data (might be tricky here), 
Mario | Increase the number of epochs / Reduces neurons and increase depth. ()
Pesi |
Rodrigo | add more layers?
Ronald | add some noise to prevent overlearning
Sam |  Train for more epochs, same nr of observations for each class, more data
Shengnan | normalize the input features, introduce validation data, models ensumble
Sobhan |
Thang |cross validation? 
Tugce |
Xiao |
Yakob |
Xuetong |

## 🧠 Collaborative Notes

### Classification by a Neural Network using Keras
Start your jupyter environment (or some other interactive Python)
```python=
jupyter lab
```

Import the required packages
```python=
from tensorflow import keras
```

1. Formulate / Outline the problem
We will train a neural network to classify penguins species. 🐧🐧🐧

2. Identify inputs and outputs
Load the dataset using seaborn
```python=
import seaborn as sns
penguins = sns.load_dataset('penguins')
penguins
```

Exercise: see line 212 of this document

Let's visualize the data, where each species has a different colour
```python=
sns.pairplot(penguins, hue='species')
```

We will remove the island and sex features, and only work with the numerical features. We will also drop the species columns from the features, as it is our target class so we shouldn't use it as input feature.
```python=
penguin_features = penguins.drop(columns=['species', 'island', 'sex'])
target = penguins['species']
penguin_features.columns
```

3. Prepare data
We will now prepare our data to be used for training a neural network.

The species column has to be converted to something numerical so the neural network can read it. First we will convert the column to a categorical feature.

```python=
target = target.astype('category')
target
```
The dtype has changed to "category".

Next, we have to deal with missing values (NaN) in the data. We will just remove the rows that contain a NaN in any column.
```python=
penguins_filtered = penguins.drop(columns=['island', 'sex']).dropna()
```
We have to redefine the features and targets, as they were previously created from the data that still contained NaNs
```python=
penguin_features = penguins_filtered.drop(columns='species')
target = penguins_filtered['species'].astype('category')
target.describe()
```
There are now 342 samples left, so 2 were removed.

We will convert the species to numerical values. This is done using something called one-hot encoding, which is built into pandas.
```python=
import pandas as pd

target = pd.get_dummies(target)
target.head()
```

The data now look ok, but have to be split into a train and test set. There is 
a function for this in scikit-learn
```python=
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(penguin_features, target, test_size=.2, 
                     random_state=0, shuffle=True, stratify=target)
```
```python=
len(X_train)
```
```python=
len(X_test)
```
The training set has 273 samples, the test set has 69.

Are the classes in the training and testing set well balanced?
```python=
y_train.value_counts()
```
The original data and testing set have a similar class balance
```python=
y_test.value_counts()
target.value_counts()
```

The classes are not completely balanced, but not extremely unbalanced so we will leave them as-is. 


4. Build a neural network
```python=
from tensorflow import keras
```
To make sure everyone has (nearly) the same results, we will set some random seeds
```python=
from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
```

We will not use a pre-trained network, but build an architecture from scratch.
The first hidden layer will be a dense layer, which means that all neurons of the layer are connected to all inputs and outputs of the layer.

We first create an input layer, which we need to tell how many inputs (= features) we have
```python=
print(X_train.shape)  # (273, 4): 273 samples and 4 features
inputs = keras.Input(shape=X_train.shape[1])
```

Next, we add the dense hidden layer with 10 neurons (at this point, this number is just a guess)
```python=
hidden_layers = keras.layers.Dense(10, activation='relu')(inputs)
```
The relu function is defined as relu(x) = max(0, x)

Finally, we add an output layer with 3 neurons (one for each species)
```python=
output_layer = keras.layers.Dense(3, activation='softmax')(hidden_layers)
```
The softmax activation function normalizes the output to sum to 1, and each output will be in the 0-1 range.

Now we define a model that uses these layers
```python=
model = keras.Model(inputs=inputs, outputs=output_layer)
model.summary()
```

Exercise: see line 242

5. Choose a loss function and optimizer
To be able to train the model we need to choose a loss function and optimizer.
They are used to assess how well the model is doing, and change the weights to make it (hopefully) perform better.

A typical loss function for comparing categorical output to the true category is categorical cross-entropy loss. We will use the Adam optimizer. The optimizer decides how much the weights and biases should be changed given the output of the loss function.

```python=
model.compile(optimizer=keras.optimizers.Adam(), 
              loss=keras.losses.CategoricalCrossentropy())
```

6. Train model
Now we can finally train the network
```python=
history = model.fit(X_train, y_train, epochs=100)
```

We can plot the training process
```python=
sns.lineplot(x=history.epoch, y=history.history['loss'])
```

Exercise: see line 279

7. Perform a prediction / classification
To assess the performance of the model, we will run the prediction on the test set.
```python=
y_pred = model.predict(X_test)
prediction = pd.DataFrame(y_pred, columns=target.columns)
prediction
```

```python=
predicted_species = prediction.idxmax(axis='columns')
predicted_species
```

8. Measuring performance
To compare the predicted species to the real ones, we will generate a confusion matrix
```python=
from sklearn.metrics import confusion_matrix

true_species = y_test.idxmax(axis='columns')
matrix = confusion_matrix(true_species, predicted_species)
matrix
```

This is a bit hard to read, so let's convert it to a dataframe and create a nicer plot
```python=
confusion_df = pd.DataFrame(matrix, index=y_test.columns.values, 
                            columns=y_test.columns.values)

confusion_df.index.name = 'True label'
confusion_df.columns.name = 'Predicted label'

sns.heatmap(confusion_df, annot=True)
```

The model does not perform well on the test set (yet!).

Exercise: see line 309

9. Tune hyperparameters
There are many hyperparameters to choose from. These are the decision we made when generating the model: the number of layers, number of neurons in each layer, which loss function we use etc. We will discuss this more later in the workhop. For now it is important to realize that the parameters we chose were somewhat arbitrary and more careful consideration needs to be taken to pick hyperparameter values.

10. Share model
As a last step, we will save the model so it can be reused later.
```python=
model.save('my_first_model')
```

You can then later load it with
```python=
pretrained_model = keras.models.load_model('my_first_model')
pretrained_model.predict(X_test)
```

### Tips
- Include also the recommended python version, and maybe share a conda environment .yml file to download :+1: :+1: :+1: :+1:
- A short presentation on the structure of a neural net, where activation function and bias are introduced, would be helpful :+1: :+1: :+1:  :+1:
- I followed the setup instructions which worked out, but my setup was not correct in the end. Perhaps include a better test if the setup is correct?
- Perhaps provide bash/batch script to help people setup their environments
- Suggest pycharm for the setup, as it manages env well and avoids windows v.s. linux v.s. mac issues. 
- A bit more in depth on how NN work under the hood, as opposed to immeadiatly using a library :+1:
- I did not get the meeting link in the begining :( 
- More consistency in variable naming (i.e. `penguins_filtered` and `penguin_features`; `input` and `hidden_layer`)
- When steps are inserted within earlier code a bit more explaining on the reason why and the implication for the existing code (i.e. `penguins_filtered`)
- At the start of the course the pace was relaxed but at the moment the actual coding for DL started the pace was very high. This might be better balanced.

### Tops
- Very clear structure to follow, thanks!
- Love the collaborative document! Saved me a lot of time making notes and allows me to focus on the course :+1: :+1:
- The helpers in the chat to keep people "on track" during the course were great :+1: 
- Pace is really good, easy to follow :+1:
- Nice and easy to follow format
- Like the way giving the exercises during the course :+1:
- I always learn something new from this type of workshops. It is interesting to see how other collegues program! :+1: :+1:
- Thanks for nicely structuring the tutorial. Very clear.
- Nothing to add, easy to follow and very clear (:
- Nice and very clear
## :car: Parking Lot

## 📚 Resources
[NL-RSE](https://www.nl-rse.org)
[Next NL-RSE meetup](https://www.eventbrite.co.uk/e/nl-rse-meetup-software-development-in-industry-vs-academia-tickets-195099246097)
[Tensorflow playground](https://playground.tensorflow.org)
[Netherlands eScience Center newsletter](https://esciencecenter.us8.list-manage.com/subscribe/post?u=a0a563ca342f1949246a9f92f&id=31bfc2303d)
[Google Colab online environment](https://colab.research.google.com/)
[MIT lecture about neurons, layers, and bias](https://www.youtube.com/watch?v=njKP3FqW3Sk&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=13&ab_channel=AlexanderAmini)
[Loss functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses)