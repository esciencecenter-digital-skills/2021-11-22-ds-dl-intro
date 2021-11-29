![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Deep Learning Day 3

2021-11-24 Introduction Deep Learning

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

Post-workshop survey

[link](https://www.surveymonkey.com/r/7VQCMNT)

## 👩‍🏫👩‍💻🎓 Instructors

Djura Smits, Sven van der Burg

## 🧑‍🙋 Helpers

Yang Liu, Leon Oostrum

## 🗓️ Agenda
| | |
|-|-|
|09:00|Welcome and icebreaker|
|09:15|Networks are like onions|
|10:15|Coffee break|
|10:30|Networks are like onions|
|11:45|Tea break|
|12:00|Networks are like onions|
|12:30|Q&A|
|12:45|Post-workshop survey|
|13:00|END|

## 🔧 Exercises

### Continued from yesterday: Monitoring the training process
#### Open question: What could be next steps to further improve the model? (breakout rooms)
With unlimited options to modify the model architecture or to play with the training parameters, deep learning can trigger very extensive hunting for better and better results. Usually models are “well behaving” in the sense that small changes to the architectures also only result in small changes of the performance (if any). It is often tempting to hunt for some magical settings that will lead to much better results. But do those settings exist? Applying common sense is often a good first step to make a guess of how much better could results be. In the present case we might certainly not expect to be able to reliably predict sunshine hours for the next day with 5-10 minute precision. But how much better our model could be exactly, often remains difficult to answer.

- What changes to the model architecture might make sense to explore?
- Ignoring changes to the model architecture, what might notably improve the prediction quality?

Answers:
- Room 1:
- Room 2: 
    - Increase no. of layers (implement (interaction of) many parameters)
    - add noise to parameters 
    - hyperparameter tunning
    - normalization of output to bring all the y's into the range of [0,1]
    - increase the weight of input data from cities nearby (include proximity)
- Room 3:
    - Increase or decrease the number of layers
    - Change the activation functions
    - Pre-processing of the data. Perhaps it could be cleaned up further.
    - Ideally we would have more data?
    - Maybe this is not a good task for NN?
- Room 4: 
    - More layers, different activation or loss functions, less neurons to reduce number of parameters
    - Reshape the data: make each city its own entry so the model can learn about the relationships between humidity and sunshine, rather than maybe sunshine in location A and sunshine in location B
    - Normalize or scale features, use more training data by using nested cross-validation
- Room 5:
    - More layers
    - Less Dense Layers
    - More/Less node
    - Ensembl model?

    - Change input structure: More previous days to predict trends
    - Get more representative data (More data), more features

#### Exercise: Explore the data

 Familiarize yourself with the CIFAR10 dataset. To start, consider the following questions:
 - What is the dimension of a single data point? What do you think the dimensions mean?
 - What is the range of values that your input data takes?
 - What is the shape of the labels, and how many labels do we have?
 - (optional): Try to follow the first few steps of the deep learning workflow: 1) Formulate the problem, 2) identify inputs and outputs. Then think about step 4: see if you can find a pre-trained model or architecture that is known to work on this dataset.

Write your answers here:
| Name |  |
|------|--|
Artur | 32 x 32 x 3, range is 0-255, labels are integers with range 0-9. We have a classification problem that takes images as input and outputs labels. Image classification is not possible with a "regular" neural net because of the input dimensions, you need a convolutional neural net instead.
Cristian |
Dimitrios | 32x32x3,,RGB-pixels, Range: 0-255,  dtype=uint8, 10 labels
Duong | 
Felicia | 32 x 32 x3 (dpi), Label shape: [50000, ],  [0 - 9] 10 diff. labels 
Florian | 32,32,3, pixel rgb values, range:: 0-255, 10 labels represented as ints, 
Jelle | 32, 32, 3; 0-255; 10 labels (0-9)
Khun Sang |dimension: 3, range: 0-255, shape: (32,32,3)
Kumah |
Leonoor | 
Lukasz | 1) (32, 32, 3); 2) 0-255; 3) (5000, 1) in range (0, 9)
Pesi | 1) shape 32,32, 3 , 2)RGB images 
Rodrigo | 1: (32, 32, 3), pixels? 2: 0-255 3: (50000, 1) (or 5000 in our subset) 4: 
Ronald | images: 32x32 RGB(?) pixels, (uint8) 0-255; labels: 1, (uint8) 0-9; problem: catagorize 32x32x3 images to one of 10 labels; dimensions: 3072 input nodes, 9 binary output nodes; approach: start with same network as penguins-problem
Sam | 32x32x3 (prob RGB low resolution image) - 0-255 (int8) - 1, - Classify the object in small images. Input are 3 coloured 32x32 images (So 3072 input nodes), output should be the probability of all 10 classes (Softmax).
Sobhan | 32, 32, 3, range: 0 - 255
Thang | (32, 32, 3) 
Tugce |

Solution:
```python=
train_images.shape
```
A single datapoint has dimensions of (32, 32, 3). This is a 32 by 32 pixel image with 3 colour channels (RGB).

```python=
train_images.min(), train_images.max()
```
The data ranges from 0 to 255 (integer-based).

```python=
train_labels.shape
```
There are 5000 samples and one label per sample.
```python=
train_labels.min(), train_labels.max()
```
The labels range from 0 to 9, so there are 10 different labels (10 classes).

#### Exercise: Convolutional neural networks
In breakout rooms, answer the following questions. Write your answers in the collaborative document.

##### 1. Border pixels
What, do you think, happens to the border pixels when applying a convolution?
##### 2. Number of parameters
Suppose we apply a convolutional layer with 100 kernels of size 3 * 3 * 3 (the last dimension applies to the rgb channels) to our images of 32 * 32 * 3 pixels. How many parameters do we have? Assume, for simplicity, that the kernels do not use bias terms. Compare this to the number of parameters required for a dense layer (307300).
##### 3. A small convolutional neural network
So let’s look at a network with a few convolutional layers. We need to finish with a Dense layer to connect the output cells of the convolutional layer to the outputs for our classes.
```python=
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model.summary()
```
Inspect the network above:

* What do you think is the function of the Flatten layer?
* Which layer has the most parameters? Do you find this intuitive?

Anwers:
- Room 1
    - .
    - .
- Room 2
    1. zeros padding, circular padding (lines at the bottom goes to row -1), mirroring
    2. `100*(3*3*3) = 2700`
    3. Flatten layer function: Translation into classes (in shape of a vector)
- Room 3:
    - 1. We should probably pad our images so we obtain consistent shape of the output. WE could use zero padding or constant padding.
    - 2. If the kernel is 3 x 3 x 3 then we have 27 parameters per kernel and we have 100 kernels so its 27 * 100 = 2700.
    - 3. 
        - Flatten -> will turn 2d/3d matrix into a 1d vector
        - The last layer (Dense) will have the most parameters since its creating large number of connects. All inputs to all outputs.
- Room 4: 
    1. If we assume that the subject is somewhat centered, optimal kernels will set the border pixels to 0.
    2. 3x3x3x100 = 2700 parameters if we ignore the bias term. This is a much smaller number of parameters.
    3. Flatten reduces the number of dimensions to 1.
    4. The output layer has the most parameters.
- Room 5: 
    - 1. Border pixels will most likely be padded by zeroes, can also be mirrored?
    - 2. 3*3*3*100  = 2700
    - 3. 
        - Reduce dimensionality of input data, create 1D array out of 2D image
        - The last dense layer


Solution:
1. There are several options for the border, padding, ignore the borders, copy the first/last row. The Keras default is to ignore the borders (so the output of the layer is smaller than the input)
2. The number of parameters only depends on the kernels: 3 * 3 * 3 * 100 = 2700
3. The flatten layers turns the multi-dimensional output of the convolutional layer into a single 1D vector which is required as input to the dense layer.  The dense layer has the most parameters, by far. 

#### Exercise: Network depth
What, do you think, will be the effect of adding a convolutional layer to your model? Will this model have more or fewer parameters? Try it out. Create a model that has an additional Conv2d layer with 32 filters after the last MaxPooling2D layer. Train it for 20 epochs and plot the results.

Write down your answers in the collaborative notebook.

HINT: The model definition that we used previously needs to be adjusted as follows:
```python=
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
# Add your extra layer here
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32)(x)
outputs = keras.layers.Dense(10)(x)
```
Optional: Browse the web for existing architectures or pre-trained models that are likely to work well on this type of data. The extra layer does not improve the results (much).

Answers:
| Name |  |
|------|--|
Artur | Because of the extra pooling layer, we have fewer parameters which should help reduce overfitting. The validation accuracy seems similar.
Cristian |
Dimitrios | Total params: 24,522, Accuracy, loss graphs are similar with the previous model, performance in train set slightly better
Duong | 
Felicia | Fewer params 24,522 (without 2nd pooling layer 58,122) Val-Accuracy old model: 0.5277; model with additional layer: 0.5216 (slightly lower). By using only 10 epochs (like in previous model): 0.4982
Florian | less params, accuracy is lower but comparable
Jelle |
Khun Sang | Accuracy is lower, ess parameters, original model: 50K, model+convlayer+pooling:23850
Kumah |
Leonoor |
Lukasz | Original model: 21,674 parameters; Extra Conv layer: 24,522. Both have similar accuracy of 0.7
Pesi | fewer parameters, acc-ori: 0.6314, add conv extra: 0.7194 , add pooling extra: 0.8792
Rodrigo
Ronald | accuracy is lower
Sam | Original model: 21674 pars, Model+32*Convlayer: 24522 pars, Model+Conv+pooling: 20682. Train accurarcy seems higher, Test accuracy remains similar
Sobhan |
Thang | fewer parameters
Tugce |less parameters

Solution:
Adding the extra layer:
```python=
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3,3), activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32)(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='cifar_model_extra_layer')
model.summary()
```
If you have the 32-neuron dense layer, there are actually fewer parameters with the extra convolution layer, because the dense layer will be smaller.

## 🧠 Collaborative Notes

### Networks are like onions

We will discuss:

- Why do we need different types of layers?
- What are good network designs for images data?
- What is a convolutional layers?
- How can we avoid overfitting?

We will work on a new dataset called CIFAR10
```python=
from tensorflow import keras
(train_images, train_labels), (test_images, test_labels) = \
    keras.datasets.cifar10.load_data()

len(train_images)
```

To keep things fast enough, we will only use the first 5000 images
```python=
n = 5000
train_images = train_images[:n]
train_labels = train_labels[:n]
```

Step 0: inspect the data
Exercise on line 155

#### Code for plotting images
```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.axis('off')
    plt.title(class_names[train_labels[i,0]])
plt.show()
```

Step 1: Formulate the problem
Classify images, predict given an images which class it is out of the 10 available classes

Step 2: Inputs and outputs
Inputs: images
Outputs: classes

Step 3: Prepare data
Neural networks like small numbers, so we divide the images by the maximum value
```python
train_images = train_images / 255.
test_images = test_images / 255.

train_images.min(), train_images.max()
```

Step 4: Choose a pre-trained model or build a new architecture from scratch
Example of a pre-trained model on CIFAR10: [link](https://github.com/Adeel-Intizar/CIFAR-10-State-of-the-art-Model)
For this workshop however, we will build our own network.

If we would use dense layers, the images would effectively be flattened into one long array of pixel values. How many points would there be?
```python=
dim = train_images.shape[1] * train_images.shape[2] * train_images.shape[3]
```

Convolutional layers apply a (typically small) kernel to an image.
They can take into account the geometric relation between pixels, and require
fewer parameters than a dense layer: the values in the e.g. 3x3 kernel are the parameters.

Exercise on line 205

Pooling layers are another type of layer. The MaxPooling2D layer takes the maximum pixel value in each block of nxm pixels, where the n and m are arguments of the layer. In this example, we take the maximum in each 2x2 pixel block.
We will add one after each convolutional layer (original code without pooling from exercise on line 205):
```python=
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_small")

model.summary()
```

Step 5: Choose a loss function and optimizer
```python=
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
Note that we didn't do anything with our labels: they are not one-hot encoded. That is why we choose the _sparse_ CategoricalCrossentropy. from_logits=True is chosen because the model does not include e.g. a softmax activation in the output layer. from_logits does the scaling to a probality value for us.

Step 6: Training the model
```python=
history = model.fit(train_images, train_labels,
                    epochs=10,
                    validation_data=(test_images, test_labels))
```

Let's plot the training process
```python=
import seaborn as sns
import pandas as pd

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
sns.lineplot(data=history_df[['loss', 'val_loss']])
```

Exercise on line 262

We now skip the remaining steps of the workflow and continue with a new type of layer:

Dropout layers are meant to reduce overfitting. They randomly disable a fraction of neurons during training, but _not_ when doing predictions.

```python=
inputs = keras.Input(shape=train_images.shape[1:])
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
# Add dropout layer
x = keras.layers.Dropout(.2)(x)
x = keras.layers.Flatten()(x)
# Added dense layer
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="cifar_model_with_dropout")

model.summary()
```

Let's compile and train the model using the same functions as before
```python=
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels,
                    epochs=20,
                    validation_data=(test_images, test_labels))

history_df = pd.DataFrame.from_dict(history.history)
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']])
sns.lineplot(data=history_df[['loss', 'val_loss']])
```

We are still overfitting somewhat, but the gap between the training and validation data is now smaller. You could further try to improve by e.g. increasing the dropout fraction.

### Recap & Q&A
Take a few minutes to write down your thoughts on what we learned about deep learning in this course:
* What questions do you still have?
* Whether there are any incremental improvements that can benefit your projects?
* What’s nice that we learnt but is overkill for your current work?

| Name |  |
|------|--|
Artur |
Cristian |
Dimitrios | I could follow most of the subjects even though I don't have an intense background on CS, and I realised I have to dive deeper to the theory of the material shown. I currently work with the identification of DNA regulatory elements, and I would like to test if a DL approach would work proprly on my case. The content of the workshop is optimised to kickstart someone on how to use DL approaches, so no overkill detected. 
Duong | 
Felicia | Workflow, applications (problems that can be solved), interpret the improvement for different adjustments. In my side-project, I want to build up a model to classify different forms of herbivory/health status of plants in the field - digital phenotyping (agricultural robotics), might be also interesting to identify species. Used 'classical' ML before for this task. This was my first excursion into DL, and it was really helpful as an introduction. I will continue learning about keras. 
Florian | More technical details
Jelle | How would you go about using a pretrained model for a slightly different situation? We want to classify monkey sounds in the jungle, could we use a pretrained model that classifies other sounds ? The course is really useful, I can directly apply one of the Keras examples in our project. Im a little bit affraid of installation issues to make it run on my GPU. No overkill.
Khun Sang | I can apply some of skills learned in this workshop in my work. I am looking forward in using DL for Bayesian inference. 
Kumah |
Leonoor | How do you do transfer learning? Also, I am a bit confused about batch normalization, as opposed to preprocessing normalization. 
Lukasz | 1) I am still not entirely sure how I could apply NN to my own problem but I haven't quite figured out how to convert it from traditional approaches to deep learning task. 2) 
Pesi |
Rodrigo | Quite a lot of questions still, a lot more to learn. (: I'm not working on anything deeplearning at the moment, took this as an introduction to see if I could apply it somewhere.
Ronald | 
Sam | Is there a way to beforehand decide how much input data will most likely be required? How can i keep the NN from burning my CPU..? I will defintely be able to implement the stuff I learned in some projects I am doing, just not worth it for super easily answerable questions. 
Sobhan |
Thang | How do we share/publish trained models? how do we use pretrained models as transfer learning?
Tugce |I enjoy the workshop and it's a good start for me to learn NN. 


### Tips
- 

### Tops
- 

## :car: Parking Lot
* It would be nice to have a guide of improving the model like a flowchart. If X do Y, etc :+1: :+1:
* How to decide number of neurons or layers?

Is there a way to beforehand decide how much input data will most likely be required?
* You can try to find it in literature. If you have a small dataset you can do a sensitivty analysis: https://machinelearningmastery.com/sensitivity-analysis-of-dataset-size-vs-model-performance/

How would you go about using a pretrained model for a slightly different situation? We want to classify monkey sounds in the jungle, could we use a pretrained model that classifies other sounds ?
* This is exactly the right usecase for transfer learning (I would define transfer learning as pretraining on a slightly different domain and then applying that to your own domain). The intuition would be that the model learns something generic about sounds by pretraining on a larger sound classifcation dataset. 

How do we share/publish trained models?
* Add to a model zoo
* Put your code on github, add requirements.
* Use Keras to save to hdf5, store in zenodo (for example: https://zenodo.org/record/4699356)


## 📚 Resources
[post-workshop survey](https://www.surveymonkey.com/r/7VQCMNT)
[CIFAR10 state of the art model in Keras](https://github.com/Adeel-Intizar/CIFAR-10-State-of-the-art-Model)
[MIT lecture on convolutional networks](https://www.youtube.com/watch?v=iaSUYvmCekI&ab_channel=AlexanderAmini)
[Explainable AI software](https://github.com/dianna-ai/dianna)
[Andrew Ng's book](https://www.goodreads.com/en/book/show/30741739-machine-learning-yearning)
[NL-RSE mailing list](https://nl-rse.org/pages/join)