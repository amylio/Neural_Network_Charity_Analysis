# Neural_Network_Charity_Analysis

An exercise in Neural Networks and Deep Learning Models using TensorFlow and Pandas libraries in Python to preprocess datasets and create a predictive binary classifier.

## Overview

The purpose of this analysis was to explore and implement neural networks using `TensorFlow` in `Python`. Neural networks is an advanced form of `Machine Learning` that can recognize patterns and features in the dataset. Neural networks are modeled after the human brain and contain layers of neurons that can perform individual computations. A great example of a `Deep Learning Neural Network` would be image recognition. The neural network will compute, connect, weigh and return an encoded categorical result to identify if the image represents a "dog" or a "cat" or as shown in the image below, George Washington. 

![neural](https://github.com/amylio/Neural_Network_Charity_Analysis/blob/main/Images/Neural.jpg)

Throughout this module, we learned:

* How to build a basic neural network
* Preprocess/prepare the datasets
* Create a training and testing set
* Measure model accuracy
* Add additional neurons and hidden layers to optimize the model
* Select the best model to use for our dataset

**AlphabetSoup**, a philanthropic foundation is requesting for a mathematical, data-driven solution that will help determine which organizations are worth donating to and which ones are considered "high-risk". In the past, not every donation AlphabetSoup has made has been impactful as there have been applicants that will receive funding and then disappear. **Beks**, a data scientist for AlphabetSoup is tasked with analyzing the impact of each donation and vet the recipients to determine if the company's money will be used effectively. In order to accomplish this request, we are tasked with helping Beks create a binary classifier that will predict whether an organization will be successful with their funding. We utilize `Deep Learning Neural Networks` to evaluate the input data and produce clear decision making results.

## Results

We used a CSV file containing more than 34,000 organizations that have received past contributions over the years. The following information was contained within this dataset.

CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively

### Data Preprocessing

To start, we needed to preprocess the data in order to compile, train and evaluate the neural network model. For the ***Data Preprocessing*** portion:

* **EIN** and **NAME** columns were removed during the preprocessing stage as these columns added no value.
* We also binned **APPLICATION_TYPE** and categorized any unique values with less that 500 records as "Other"  
* **IS_SUCCESSFUL** column was the target variable.
* The remaining 43 variables were added as the features (i.e. STATUS, ASK_AMT, APPLICATION TYPE, etc.)

### Compiling, Training and Evaluating the Model

After the data was preprocessed, we used the following parameters to ***compile, train, and evaluate the model***:

* The initial model had a total of 5,981 parameters as a result of 43 inputs with 2 hidden layers and 1 output layer. 
  * The first hidden layer had 43 inputs, 80 neurons and 80 bias terms. 
  * The second hidden layer had 80 inputs (number of neurons from first hidden layer), 30 neurons and 30 bias terms.
  * The output layer had 30 inputs (number of neurons from the second hidden layer), 1 neuron, and 1 bias term. 
  * Both the first and second hidden layers were activated using `RELU - Rectified Linear Unit` function. The output layer was activated using the `Sigmoid` function. 

* The target performance for the accuracy rate is greater than 75%. The model that was created only achieved an accuracy rate of 72.33%

![orig](https://github.com/amylio/Neural_Network_Charity_Analysis/blob/main/Images/Orig2.png)

### Attempts to Optimize and Improve the Accuracy Rate

Three additional attempts were made to increase the model's performance by changing features, adding/subtracting neurons and epochs. The results did not show any improvement. 

![results](https://github.com/amylio/Neural_Network_Charity_Analysis/blob/main/Images/Results.png)

  * **Optimization Attempt #1:**
    * Binned **INCOME_AMT** column
    * Created 5,821 total parameters, an decrease of 160 from the original of 5,981
    * Accuracy improved 0.13% from 72.33% to 72.42%
    * Loss was reduced by 2.10% from 58.08% to 56.86%

  * **Optimization Attempt #2:**
    *  Removed **ORGANIZATION** column
    *  Binned **INCOME_AMT** column
    *  Removed **SPECIAL_CONSIDERATIONS_Y** column from features as it is redundant to **SPECIAL_CONSIDERATIONS_N**
    *  Increased neurons to 100 for the first hidden layer and 50 for the second hidden layer
    *  Created 8,801 total parameters, an increase of 2,820 from the original of 5,981
    *  Accuracy decreased 0.11% from 72.33% to 72.24%
    *  Loss increased by 1.75% from 58.08% to 59.10%
    
  * **Optimization Attempt #3:**
    *  Binned **INCOME_AMT** and **AFFILIATION** column
    *  Removed **SPECIAL_CONSIDERATIONS_Y** column from features as it is redundant to **SPECIAL_CONSIDERATIONS_N**
    *  Increased neurons to 125 for the first hidden layer and 50 for the second hidden layer
    *  Created 11,101 total parameters, an increase of 5,120 from the original of 5,981
    *  Accuracy increased 0.19% from 72.33% to 72.47%
    *  Loss decreased by 1.82% from 58.08% to 57.02%

## Summary

In summary, our model and various optimizations did not help to achieve the desired result of greater than 75%. With the variations of increasing the epochs, removing variables, adding a 3rd hidden layer (done offline in Optimization attempt #4) and/or increasing/decreasing the neurons, the changes were minimal and did not improve above 19 basis points. In reviewing other `Machine Learning` algorithms, the results did not prove to be any better. For example, `Random Forest Classifier` had a predictive accuracy rate of 70.80% which is a 2.11% decrease from the accuracy rate of the `Deep Learning Neural Network` model (72.33%). 

Overall, Neural Networks are very intricate and would require experience through trial and error or many iterations to identify the perfect configuration to work with this dataset.

## Resources
* **Software:** Python 3.7.9, Anaconda 4.9.2 and Jupyter Notebooks 6.1.4
* **Libraries:** Scikit-learn, Pandas, TensorFlow, Keras
