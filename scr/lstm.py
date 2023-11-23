import numpy as np

# Inicjalizacja wag sieci
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))

# Klasa LSTM
class LSTM:
    # Inicjalizacja sieci LSTM
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hiperparametry
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        # Inicjalizacja wag i biasów dla różnych bramek LSTM
        # 1 Forget Gate
        self.wf = initWeights(input_size, hidden_size)
        self.bf = np.zeros((hidden_size, 1))

        # 2 Input Gate
        self.wi = initWeights(input_size, hidden_size)
        self.bi = np.zeros((hidden_size, 1))

        # 3 Cell State Gate
        self.wc = initWeights(input_size, hidden_size)
        self.bc = np.zeros((hidden_size, 1))

        # 4 Output Gate
        self.wo = initWeights(input_size, hidden_size)
        self.bo = np.zeros((hidden_size, 1))



    

