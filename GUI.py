# The PySimpleGUI used for formatting and creating the user interface
import PySimpleGUI as sg
# My main file with the neural network code
import main
# numpy import again
import numpy as np
# time for timing the program speed
import time

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
    [sg.Button("Start AI", button_color=('white', 'green'), enable_events=True, key='STAI')],
    [sg.Text("Answers: ")],
    [sg.Text("2: ")], [sg.Button("Yes(2)", button_color=('white', 'green'), enable_events=True, key='Y2'),
                       sg.Button("No(2)", button_color=('white', 'green'), enable_events=True, key='N2')],
    [sg.Text("3: ")], [sg.Button("Yes(3)", button_color=('white', 'green'), enable_events=True, key='Y3'),
                       sg.Button("No(3)", button_color=('white', 'green'), enable_events=True, key='N3')],
    [sg.Text("4: ")], [sg.Button("Yes(4)", button_color=('white', 'green'), enable_events=True, key='Y4'),
                       sg.Button("No(4)", button_color=('white', 'green'), enable_events=True, key='N4')],
    [sg.Text("5: ")], [sg.Button("Yes(5)", button_color=('white', 'green'), enable_events=True, key='Y5'),
                       sg.Button("No(5)", button_color=('white', 'green'), enable_events=True, key='N5')],
    [sg.Text("6: ")], [sg.Button("Yes(6)", button_color=('white', 'green'), enable_events=True, key='Y6'),
                       sg.Button("No(6)", button_color=('white', 'green'), enable_events=True, key='N6')],
    [sg.Text("7: ")], [sg.Button("Yes(7)", button_color=('white', 'green'), enable_events=True, key='Y7'),
                       sg.Button("No(7)", button_color=('white', 'green'), enable_events=True, key='N7')],
    [sg.Text("8: ")], [sg.Button("Yes(8)", button_color=('white', 'green'), enable_events=True, key='Y8'),
                       sg.Button("No(8)", button_color=('white', 'green'), enable_events=True, key='N8')],
    [sg.Text("9: ")], [sg.Button("Yes(9)", button_color=('white', 'green'), enable_events=True, key='Y9'),
                       sg.Button("No(9)", button_color=('white', 'green'), enable_events=True, key='N9')],
    [sg.Text("10: ")], [sg.Button("Yes(10)", button_color=('white', 'green'), enable_events=True, key='Y10'),
                        sg.Button("No(10)", button_color=('white', 'green'), enable_events=True, key='N10')],
    [sg.Button("Submit Answers", button_color=('white', 'green'), enable_events=True, key='SA')]
]

layout = [
    [
        sg.Column(question_column),
        sg.VSeperator(),
        sg.Column(yes_no_column)
    ]
]


def add_answer(button_number, yes_no):
    for i in range(1):
        if yes_no == "Yes":
            GUI_input.insert(button_number - 2, 1)
        else:
            GUI_input.insert(button_number - 2, 0)


def output(train):
    if train == 1:
        return "The AI thinks you may have heart disease."
    elif train == 0:
        return "The AI thinks you don't have heart disease."
    else:
        return "The AI would need more information to make a decision."


window1 = sg.Window("AI Health App", layout)
GUI_input = []
past_trained_weights = []

while True:
    event, values = window1.read()
    if event == sg.WIN_CLOSED:
        break

    if event == "STAI":
        start = time.time()
        print("Starting Timer: 0s")
        main.neural_network = main.ANN()

        print("Initial Weights: ")
        print(main.neural_network.synaptic_weights)

        training_inputs = main.train_inputs
        training_outputs = main.train_outputs

        main.neural_network.train(training_inputs, training_outputs, 15000)

        print('Weight after training: ')
        print(main.neural_network.synaptic_weights)
        past_trained_weights.append(main.neural_network.synaptic_weights)
        end = time.time()
        elapsed = end - start
        minutes = elapsed/60
        print("Program Finished in " + str(minutes) + " minutes")

    if event == "Y2":
        add_answer(2, "Yes")
    elif event == "N2":
        add_answer(2, "No")
    elif event == "Y3":
        add_answer(3, "Yes")
    elif event == "N3":
        add_answer(3, "No")
    elif event == "Y4":
        add_answer(4, "Yes")
    elif event == "N4":
        add_answer(4, "No")
    elif event == "Y5":
        add_answer(5, "Yes")
    elif event == "N5":
        add_answer(5, "No")
    elif event == "Y6":
        add_answer(6, "Yes")
    elif event == "N6":
        add_answer(6, "No")
    elif event == "Y7":
        add_answer(7, "Yes")
    elif event == "N7":
        add_answer(7, "No")
    elif event == "Y8":
        add_answer(8, "Yes")
    elif event == "N8":
        add_answer(8, "No")
    elif event == "Y9":
        add_answer(9, "Yes")
    elif event == "N9":
        add_answer(9, "No")
    elif event == "Y10":
        add_answer(10, "Yes")
    elif event == "N10":
        add_answer(10, "No")

    if event == "SA":
        print(GUI_input)
        layout2 = [
            [sg.Text(output((main.neural_network.think(np.array([GUI_input])))))],
            [sg.Text("Close this window to "
                     "enter different data, "
                     "w/o running the AI "
                     "again.")]]
        window2 = sg.Window("AI App Results", layout2)
        event, values = window2.read()
        GUI_input = []
        past_trained_weights = []
