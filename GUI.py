# The PySimpleGUI used for formatting and creating the user interface
import PySimpleGUI as sg
# My main file with the neural network code
import main
# numpy import again
import numpy as np

count = 0

question_column = [
    [
        sg.Text("Health Questions: "
                "\n\n2: Do you have high blood pressure?"
                "\n\n3: Do you have high cholesterol"
                "\n\n4: Do you smoke often?"
                "\n\n5: Have you ever had a stroke?"
                "\n\n6: Do you have diabetes?"
                "\n\n7: Do you exercise often?"
                "\n\n8: Do you eat fruits often?"
                "\n\n9: Do you eat veggies often?"
                "\n\n10: Do you have regular check ups?")

    ]
]

yes_no_column = [
    [sg.Button("Start AI", enable_events=True)],
    [sg.Text("Answers: ")],
    [sg.Text("2: ")], [sg.Button("Yes(2)", enable_events=True), sg.Button("No(2)", enable_events=True)],
    [sg.Text("3: ")], [sg.Button("Yes(3)", enable_events=True), sg.Button("No(3)", enable_events=True)],
    [sg.Text("4: ")], [sg.Button("Yes(4)", enable_events=True), sg.Button("No(4)", enable_events=True)],
    [sg.Text("5: ")], [sg.Button("Yes(5)", enable_events=True), sg.Button("No(5)", enable_events=True)],
    [sg.Text("6: ")], [sg.Button("Yes(6)", enable_events=True), sg.Button("No(6)", enable_events=True)],
    [sg.Text("7: ")], [sg.Button("Yes(7)", enable_events=True), sg.Button("No(7)", enable_events=True)],
    [sg.Text("8: ")], [sg.Button("Yes(8)", enable_events=True), sg.Button("No(8)", enable_events=True)],
    [sg.Text("9: ")], [sg.Button("Yes(9)", enable_events=True), sg.Button("No(9)", enable_events=True)],
    [sg.Text("10: ")], [sg.Button("Yes(10)", enable_events=True), sg.Button("No(10)", enable_events=True)],
    [sg.Button("Submit Answers", enable_events=True)]
]

layout = [
    [
        sg.Column(question_column),
        sg.VSeperator(),
        sg.Column(yes_no_column)
    ]
]

GUI_input = []
past_trained_weights = []


def add_answer(button_number, yes_no):
    if yes_no == "Yes":
        GUI_input.insert(button_number - 2, 1)
    else:
        GUI_input.insert(button_number - 2, 0)


window1 = sg.Window("AI Health App", layout)

while True:
    event, values = window1.read()
    if event == sg.WIN_CLOSED:
        break

    if event == "Start AI":
        main.neural_network = main.ANN()

        print("Initial Weights: ")
        print(main.neural_network.synaptic_weights)

        training_inputs = main.train_inputs
        training_outputs = main.train_outputs

        main.neural_network.train(training_inputs, training_outputs, 10000)

        print('Weight after training: ')
        print(main.neural_network.synaptic_weights)
        past_trained_weights.append(main.neural_network.synaptic_weights)

    if event == "Yes(2)":
        add_answer(1, "Yes")
    elif event == "No(2)":
        add_answer(1, "No")
    elif event == "Yes(3)":
        add_answer(2, "Yes")
    elif event == "No(3)":
        add_answer(2, "No")
    elif event == "Yes(4)":
        add_answer(4, "Yes")
    elif event == "No(4)":
        add_answer(4, "No")
    elif event == "Yes(5)":
        add_answer(5, "Yes")
    elif event == "No(5)":
        add_answer(5, "No")
    elif event == "Yes(6)":
        add_answer(6, "Yes")
    elif event == "No(6)":
        add_answer(6, "No")
    elif event == "Yes(7)":
        add_answer(7, "Yes")
    elif event == "No(7)":
        add_answer(7, "No")
    elif event == "Yes(8)":
        add_answer(8, "Yes")
    elif event == "No(8)":
        add_answer(8, "No")
    elif event == "Yes(9)":
        add_answer(9, "Yes")
    elif event == "No(9)":
        add_answer(9, "No")
    elif event == "Yes(10)":
        add_answer(10, "Yes")
    elif event == "No(10)":
        add_answer(10, "No")

    if event == "Submit Answers":
        count = count + 1
        if count >= 2:
            weight = past_trained_weights
        else:
            weight = main.neural_network.synaptic_weights
        layout2 = [
            [sg.Text(str(main.neural_network.think(np.array([GUI_input]), weight)))],
            [sg.Text("Close this window to "
                     "enter different data, "
                     "w/o running the AI "
                     "again.")]]
        window2 = sg.Window("AI App Results", layout2)
        event, values = window2.read()
