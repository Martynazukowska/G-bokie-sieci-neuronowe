import numpy as np
from lstm import *
from tqdm import tqdm

##### Data #####
data = """To be, or not to be, that is the question: Whether \
'tis nobler in the mind to suffer The slings and arrows of ou\
trageous fortune, Or to take arms against a sea of troubles A\
nd by opposing end them. To die—to sleep, No more; and by a s\
leep to say we end The heart-ache and the thousand natural sh\
ocks That flesh is heir to: 'tis a consummation Devoutly to b\
e wish'd. To die, to sleep; To sleep, perchance to dream—ay, \
there's the rub: For in that sleep of death what dreams may c\
ome, When we have shuffled off this mortal coil, Must give us\
 pause—there's the respect That makes calamity of so long lif\
e. For who would bear the whips and scorns of time, Th'oppres\
sor's wrong, the proud man's contumely, The pangs of dispriz'\
d love, the law's delay, The insolence of office, and the spu\
rns That patient merit of th'unworthy takes, When he himself \
might his quietus make""".lower()

# Tworzenie zestawu unikalnych znaków
chars = set(data)

# Obliczanie rozmiaru danych i liczby unikalnych znaków
data_size, char_size = len(data), len(chars)

print(f'Rozmiar danych: {data_size}, Liczba unikalnych znaków: {char_size}')


# Mapowanie znaków na indeksy i odwrotnie
char_to_idx = {c:i for i, c in enumerate(chars)}
idx_to_char = {i:c for i, c in enumerate(chars)}

train_X, train_y = data[:-1], data[1:]

print()

# Initialize Network
hidden_size = 25

lstm = LSTM(input_size = char_size + hidden_size, hidden_size = hidden_size, output_size = char_size, num_epochs = 1_000, learning_rate = 0.05)
