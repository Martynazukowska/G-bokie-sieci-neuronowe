import numpy as np
from const import *

# Funkcja do one-hot encodingu znaków
def oneHotEncode(text,char_size,char_to_idx):
    output = np.zeros((char_size, 1))
    output[char_to_idx[text]] = 1

    return output

# Inicjalizacja wag sieci
def initWeights(input_size, output_size):
    return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))

# Funkcja aktywacji sigmoidalnej
def sigmoid(input, derivative = False):
    if derivative:
        return input * (1 - input)
    
    return 1 / (1 + np.exp(-input))

# Funkcja aktywacji tanh
def tanh(input, derivative = False):
    if derivative:
        return 1 - input ** 2
    
    return np.tanh(input)

# Funkcja softmax
def softmax(input):
    return np.exp(input) / np.sum(np.exp(input))

# Klasa LSTM
class LSTM:
    # Inicjalizacja sieci LSTM
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hiperparametry
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        # Inicjalizacja wag i biasów dla różnych bramek LSTM
        # f-Forget i-Input c-Cell State o-Output l-Final/Last GATES
        self.weight_f = initWeights(input_size , hidden_size)
        self.weight_i = initWeights(input_size , hidden_size)
        self.weight_c = initWeights(input_size , hidden_size)
        self.weight_o = initWeights(input_size , hidden_size)
        self.weight_l = initWeights(hidden_size, output_size)

        self.bias_f = np.zeros((hidden_size, 1))
        self.bias_i = np.zeros((hidden_size, 1))
        self.bias_c = np.zeros((hidden_size, 1))
        self.bias_o = np.zeros((hidden_size, 1))
        self.bias_l = np.zeros((output_size, 1))

    # Forward Propogation
    def forward(self, inputs):
        # Resetowanie stanu sieci
        self.concat_inputs = {}
        self.forget_gates = {}
        self.input_gates = {}
        self.candidate_gates = {}
        self.output_gates = {}
        self.outputs = {}

        self.hidden_states = {-1:np.zeros((self.hidden_size, 1))}
        self.cell_states = {-1:np.zeros((self.hidden_size, 1))}

        outputs = []
        for q in range(len(inputs)):
            # Łączenie poprzedniego stanu ukrytego (hidden_states) z bieżącym wejściem
            self.concat_inputs[q] = np.concatenate((self.hidden_states[q - 1], inputs[q]))

            # Obliczanie aktywacji bramki zapominania (Forgot Gate)
            self.forget_gates[q] = sigmoid(np.dot(self.weight_f, self.concat_inputs[q]) + self.bias_f)
            # Obliczanie aktywacji bramki wejściowej (Imput Gate)
            self.input_gates[q] = sigmoid(np.dot(self.weight_i, self.concat_inputs[q]) + self.bias_i)
            # Obliczanie kandydata na nowy stan komórki
            self.candidate_gates[q] = tanh(np.dot(self.weight_c, self.concat_inputs[q]) + self.bias_c)
            # Obliczanie aktywacji bramki wyjściowej (Final/Last Gate)
            self.output_gates[q] = sigmoid(np.dot(self.weight_o, self.concat_inputs[q]) + self.bias_o)

            # Aktualizacja stanu komórki
            self.cell_states[q] = self.forget_gates[q] * self.cell_states[q - 1] + self.input_gates[q] * self.candidate_gates[q]
            # Aktualizacja stanu ukrytego (hidden_states)
            self.hidden_states[q] = self.output_gates[q] * tanh(self.cell_states[q])

            # Obliczanie wyjściam 
            outputs += [np.dot(self.weight_l, self.hidden_states[q]) + self.bias_l]

        return outputs
    
    def train(self, inputs, labels,tqdm,char_size,char_to_idx):
        inputs = [oneHotEncode(input,char_size,char_to_idx) for input in inputs]

        for _ in tqdm(range(self.num_epochs),desc=f'{Color.GREEN}Training Progress{Color.END}'):
            predictions = self.forward(inputs)

            errors = []
            for q in range(len(predictions)):
                errors += [-softmax(predictions[q])]
                # Inkrementacja odpowiedniego elementu błędu.
                errors[-1][char_to_idx[labels[q]]] += 1
    



    

