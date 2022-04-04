# Credit to https://www.kdnuggets.com/2018/10/simple-neural-network-python.html for information on creating the Nueral Network and Sigmoid functions
# The heart disease health indicators data set is from Kaggle. It is from a phone survey, the Behavioral Risk Factor Surveillance System, run by the CDC. https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset

# numpy library used for math functions
import numpy as np
# pandas used for data set reading and writing
import pandas as pd


dataset = pd.read_csv("heart_disease_health_indicators_BRFSS2015.csv")
train_inputs = pd.DataFrame.drop(dataset, columns=['HeartDiseaseorAttack', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income', "BMI", "CholCheck", "HvyAlcoholConsump"])
train_outputs = pd.DataFrame(dataset, columns=['HeartDiseaseorAttack'])


user_choices = []


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def user_input(iterations):
    for i in range(iterations):
        user_choices.append((input("User Input" + str(i))))


class ANN():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((9, 1)) - 1

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == '__main__':
    neural_network = ANN()

