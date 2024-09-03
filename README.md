# Background

This is my first time making a neural network model. This model is designed to preform binary classification. I did my best to explain what I could. I am still googling things I am not quite familiar with. I will clarify what information needs more googling. The data set used in this code is filled with breast cancer tumor sizes and characteristics and each record states if that specific tumor is either malignant(1) or benign(0). This is the column that shows that result.

<details>
  <summary> Output Example</summary>

  | diagnosis(1=m, 0=b) |
|--------------------:|
|1                    |
|0                    |
|1                    |

</details>


# Practical Uses

## Timely Intervention

This neural network learns from historical data (features) to identify patterns that are indicative of malignant or benign tumors. Once this model is trained, it can predict the diagnosis for new patient data. This can assist doctors in early diagnosis and treatment planning by providing a probabilistic assessment of cancer presence, which could improve patient outcomes by allowing for timely intervention. 

## Medical Research and Analysis

The model can be used by researchers to analyze patterns in medical data, such as identifying features (e.g., cell size, shape, texture) are th emost indicative of cancer. This analysis can help in understanding the disease better, developing new diagnostic techniques, and refining existing ones.

## Predictive Modeling for Health Risks

Beyond cancer, similar models could be adapted to predict other binary health outcomes, such as the risk developing a certain disease (e.g. diabetes or heart disease) based on a variety of input features. Hospitals and clinics can use these models to identify high-risk patients, tailer preventative measures, and allocate resources more effectively.

## Medical Decision Support Systems

This model could be integrated into decision support systems in healthcare settings to assist practitioners in making more informed decisions. This model provides additional data points for clinicians, helping to reduce diagnostic errors and ensure a higher standard of patient care.

## Educational and Training Purposes

The code could be used in medical education to teach students and trainees about machine learning applications in healthcare. It provides hands-on experience with AI tools, preparing the next generation of healthcare professionals to work alongside advanced technology.

## Automated Medical Screening Tools

Implementing the model as part of an automated screening tool where it flags high-risk cases for futher review. This will speed up the screening process in resource-limited settings or areas with high patient volumes, ensuring timely identification of potential health issues.

## Enhancing Diagnostic Tools with Machine Learning

Existing diagnostic tools can be augmented with machine learning models to improve their accuracy and reliability. This enhances the precision of current medical tools, making them more effective and reducing false positives/negatives.

# Code Breakdown

### Import Libraries

<details>
  <summary> Example Code</summary>

  import pandas as pd
  
  from sklearn.model_selection import train_test_split
  
  import tensorflow as tf
  
</details>

'pandas' --> This is a popular Python library used for data manipulation and analysis.
'sklearn.model_selection' --> This is a module from Scikit-Learn, a machine learning library, which provides tools for splitting datasets into training and testing sets.
'tensorflow' --> An open-source machine learning library developed by Google, this includes Keras, a high-level neural network API.

### Loading the Dataset

<details>
  <summary> Example Code</summary>

  dataset = pd.read_csv('cancer.csv')
  
</details>

This line of code reads the csv file named 'cancer.csv' into a Pandas DataFrame called 'dataset'. 

### Prepare Features and Labels

<details>
<summary> Example Code</summary>

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])

y = dataset["diagnosis(1=m, 0=b)"]
  
</details>

The DataFrame 'x' contains all of the columns from 'dataset' except the column 'diagnosis(1=m, 0=b)'. This leaves x with only the feature columns used for training the model.
The DataFrame 'y' extracts the 'diagnosis(1=m, 0=b)' column from 'dataset'.

### Split the Data into Training and Testing Sets

<details>
  <summary> Example Code</summary>

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  
</details>

This function from Scikit-Learn splits the dataset into training and testing subsets. 'x_train' training set features 80% of the data by default. 'x_test' testing set features the other 20% by default. The training and testing sets for the y dataset is broken up just like the x dataset. 'test_size=0.2' Specifies that 20% of the data will be used for testing, while the remaining 80% will be used for training.

### Define the Neural Network Model

<details>
  <summary> Example Code</summary>

  model = tf.keras.models.Sequential()
  
</details>

This initializes a sequential model. A sequential model is a linear stack of layers. This is the simplest type of neural network where each layer has one input tensor and one output tensor. 

<details>
  <summary>Example of Sequential Model</summary>

  ![image](https://github.com/user-attachments/assets/9144f2f5-471d-4d78-8f25-f70d3c77a080)

</details>

### Add Layers to the Model

<details>
  <summary> Example Code</summary>

  model.add(tf.keras.layers.Dense(256, input_shape=(30,), activation='sigmoid'))

  model.add(tf.keras.layers.Dense(256, activation='sigmoid'))

  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  
</details>

#### First Hidden Layer:
Add a Dense (fully connected) layer with 256 neurons. The input_shape specifies that the input shape is a tuple with 30 elements corresponding to the 30 input features. The activatioon function used by neurons in this layer. The sigmoid function outputs a value between 0 and 1. This is suitable for binary classification. The function is shown below.

<details>
  <summary>Sigmoid Function</summary>

  ![image](https://github.com/user-attachments/assets/52a0e89a-26ed-42bd-a9c4-33aa6da8c74f)
  
</details>

Each neuron's output is calculated as the weighted sum of its inputs plus a bias term, followed by the application of an activiation function.

<details>
  <summary>Single Neuron Mathamatics</summary>

  ![image](https://github.com/user-attachments/assets/be0bec99-0a61-4907-be91-23d4d7c5b9e3)

  Where w_i are the weights, x_i are the inputs, b is the bias, and σ is the sigmoid function.

</details>

#### Second Hidden Layer

The second hidden layer is similar to the first hidden layer, the only difference is that the output from the first hidden layer, will be the input for the second hidden layer.

#### Output Layer

<details>
  <summary>Example Code</summary>

  Model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  
</details>

This output layer follows the same arithmetic as the other layers; however, this layer only has 1 neuron. The reason there is only 1 neuron is due to the fact that this model is used for binary classification tasks.

### Compile the Model

<details>
  
  <summary> Example Code</summary>

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
</details>

This code specifies that the optimizer used is called Adam (Adaptive Moment Estimation). This is a popular optimizer that combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp. It is widely used because it adapts the learning rate based on the moving averages of the gradient and squared gradient. `(This needs more googling)`

The Loss function used for training is Binary crossentropy. This measures the difference between the predicted probability and the actual label.

The last section of this code specifies the metrics to evaluate the model during training and testing. I have it set to accuracy, this measures the proportion of correct predictions.

### Train and Evaluate the Model

<details>
  <summary> Example Code</summary>

  model.fit(x_train, y_train, epochs=1000)

  model.evaluate(x_test, y_test)
  
</details>

When training the model, it will iterate over the entire training dataset 1,000 times. More epochs usually allow the model to learn better but can also lead to overfitting if too many epochs are used. Then, when evaluating the model, it returns the loss value and metric value(s) (in this case, accuracy) for the model on the testing data.

# Potential Issues

`File Not Found:` Be sure that the 'cancer.csv' file exists for this project.

`Shape Mismatch Error:` If the number of input features does not match the 'input_shape', you will get a 'ValueError'.

`Overfitting:` Training the model with too many epochs without proper regularization might lead to overfitting. This is where the model preforms well on training data but poorly on unsen data. 

`Memory Issues:` Training with large datasets or too many epochs could lead to memory errors, especially if the system lacks sufficient resources.

`Invalid Labels:` If 'y' contains values other than 0 and 1, using 'binary_crossentropy' could cause errors or lead to incorrect model behavior.
