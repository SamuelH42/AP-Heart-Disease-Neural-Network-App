# Credit to https://www.kdnuggets.com/2018/10/simple-neural-network-python.html for information on creating ANN class
# Credit to https://www.youtube.com/watch?v=z1PGJ9quPV8&list=PL3t2bW_hFraVDOxMPH4lWCUerCqG-fIJV&index=1&t=753s for information on working with pandas

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

    print("Random Weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = train_inputs
    training_outputs = train_outputs

    neural_network.train(training_inputs, training_outputs, 10000)

    print('Weight after training: ')
    print(neural_network.synaptic_weights)

    user_input(9)

    print("Considering New Situation: " + str(user_choices))
    print('Output: ')
    print(neural_network.think(np.array([user_choices])))

