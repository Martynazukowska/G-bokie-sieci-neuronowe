import numpy as np


# Klasa LSTM
class LSTM:
    # Inicjalizacja sieci LSTM
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hiperparametry
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        self.input_size = input_size
        self.output_size = output_size



    

