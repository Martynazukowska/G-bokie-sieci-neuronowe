import numpy as np
from const import *
import copy

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
class LSTM_sgdm:
    # Inicjalizacja sieci LSTM
    def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Hiperparametry
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        # Inicjalizacja wag i biasów dla różnych bramek LSTM
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

        self.v, self.s = {}, {}
        for param in ['f', 'i', 'c', 'o', 'l']:
            self.v[param] = np.zeros_like(getattr(self, f'weight_{param}'))
            self.s[param] = np.zeros_like(getattr(self, f'weight_{param}'))


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
            # Obliczanie aktywacji bramki wyjściowej 
            self.output_gates[q] = sigmoid(np.dot(self.weight_o, self.concat_inputs[q]) + self.bias_o)

            # Aktualizacja stanu komórki
            self.cell_states[q] = self.forget_gates[q] * self.cell_states[q - 1] + self.input_gates[q] * self.candidate_gates[q]
            # Aktualizacja stanu ukrytego (hidden_states)
            self.hidden_states[q] = self.output_gates[q] * tanh(self.cell_states[q])

            # Obliczanie wyjścia
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
            
            self.backward(errors, self.concat_inputs)

            

    # Backward Propogation
    def backward(self, errors, inputs):
        # Initialize weight and bias deltas
        d_weights = {k: 0 for k in ['f', 'i', 'c', 'o', 'l']}
        d_biases = {k: 0 for k in ['f', 'i', 'c', 'o', 'l']}

        beta = 0.9 

        # Inicjalizacja następnego błędu stanu ukrytego i stanu komórki
        next_error_hs, next_error_c = np.zeros_like(self.hidden_states[0]), np.zeros_like(self.cell_states[0])

        for q in reversed(range(len(inputs))):
            error = errors[q]

            # Final Gate Weights and Biases Errors
            d_weights['l'] += np.dot(error, self.hidden_states[q].T)
            d_biases['l'] += error

            # Obliczenie błędu stanu ukrytego
            error_hs = np.dot(self.weight_l.T, error) + next_error_hs

            # Output Gate Weights and Biases Errors
            error_o = tanh(self.cell_states[q]) * error_hs * sigmoid(self.output_gates[q], derivative = True)
            d_weights['o'] += np.dot(error_o, inputs[q].T)
            d_biases['o'] += error_o

            # Cell State Error
            error_c = tanh(tanh(self.cell_states[q]), derivative = True) * self.output_gates[q] * error_hs + next_error_c

            # Forget Gate Weights and Biases Errors
            error_f = error_c * self.cell_states[q - 1] * sigmoid(self.forget_gates[q], derivative = True)
            d_weights['f'] += np.dot(error_f, inputs[q].T)
            d_biases['f'] += error_f

            # Input Gate Weights and Biases Errors
            error_i = error_c * self.candidate_gates[q] * sigmoid(self.input_gates[q], derivative = True)
            d_weights['i'] += np.dot(error_i, inputs[q].T)
            d_biases['i'] += error_i
            
            # Candidate Gate Weights and Biases Errors
            error_can = error_c * self.input_gates[q] * tanh(self.candidate_gates[q], derivative = True)
            d_weights['c'] += np.dot(error_can, inputs[q].T)
            d_biases['c'] += error_can

            # Concatenated Input Error (Sum of Error at Each Gate!)
            d_z = np.dot(self.weight_f.T, error_f) + np.dot(self.weight_i.T, error_i) + np.dot(self.weight_c.T, error_can) + np.dot(self.weight_o.T, error_o)

            # Error of Hidden State and Cell State at Next Time Step
            next_error_hs = d_z[:self.hidden_size, :]
            next_error_c = self.forget_gates[q] * error_c


        for d_ in (d_weights['f'], d_biases['f'], d_weights['i'], d_biases['i'], d_weights['c'], d_biases['c'], d_weights['o'], d_biases['o'], d_weights['l'], d_biases['l']):
            np.clip(d_, -1, 1, out = d_)

        self.update_weights_sgdm(d_weights, d_biases)

        # self.weight_f += d_weights['f'] * self.learning_rate
        # self.weight_i += d_weights['i'] * self.learning_rate
        # self.weight_c += d_weights['c'] * self.learning_rate
        # self.weight_o += d_weights['o'] * self.learning_rate
        # self.weight_l += d_weights['l'] * self.learning_rate
        
        # self.bias_f += d_biases['f'] * self.learning_rate
        # self.bias_i += d_biases['i'] * self.learning_rate
        # self.bias_c += d_biases['c'] * self.learning_rate
        # self.bias_o += d_biases['o'] * self.learning_rate
        # self.bias_l += d_biases['l'] * self.learning_rate


        # Test
    def test(self, inputs, labels,idx_to_char,char_size,char_to_idx):
        accuracy = 0
        output = ''
        probabilities = self.forward([oneHotEncode(input,char_size,char_to_idx) for input in inputs])

        for q in range(len(labels)):
            prediction = idx_to_char[np.random.choice([*range(char_size)], p = softmax(probabilities[q].reshape(-1)))]
            output += prediction
            if prediction == labels[q]:
                accuracy += 1

        print(f'Początkowa wartość : \n')
        print(f'\t {labels} \n')
        print(f'Predykcja :\n \t {"".join(output)}\n')
        
        print(f'Accuracy: {round(accuracy * 100 / len(inputs), 2)} %')

    # Funkcja aktualizacji wag przy użyciu SGDM
    def update_weights_sgdm(self, d_weights, d_biases, beta=0.2):
        for param in d_weights.keys():
            self.v[param] = beta * self.v[param] + (1 - beta) * d_weights[param]
            weight = getattr(self, f'weight_{param}')
            weight += self.learning_rate * self.v[param]
            setattr(self, f'weight_{param}', weight)
            
            bias = getattr(self, f'bias_{param}')
            bias += self.learning_rate * d_biases[param]
            setattr(self, f'bias_{param}', bias)





    

