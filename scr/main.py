import numpy as np
from lstm import *
from tqdm import tqdm

data = "Moim zdaniem to nie ma tak, że dobrze albo że nie dobrze.\
    Gdybym miał powiedzieć, co cenię w życiu najbardziej, powiedziałbym, że ludzi. Ekhm...\
    Ludzi, którzy podali mi pomocną dłoń, kiedy sobie nie radziłem, kiedy byłem sam. I co ciekawe, \
    to właśnie przypadkowe spotkania wpływają na nasze życie. Chodzi o to, \
    że kiedy wyznaje się pewne wartości, nawet pozornie uniwersalne, bywa, \
    że nie znajduje się zrozumienia, które by tak rzec, które pomaga się nam rozwijać.\
    Ja miałem szczęście, by tak rzec, ponieważ je znalazłem. I dziękuję życiu. \
    Dziękuję mu, życie to śpiew, życie to taniec, życie to miłość. \
    Wielu ludzi pyta mnie o to samo, ale jak ty to robisz?, skąd czerpiesz tę radość?\
    A ja odpowiadam, że to proste, to umiłowanie życia, to właśnie ono sprawia, \
    że dzisiaj na przykład buduję maszyny, a jutro... kto wie, \
    dlaczego by nie, oddam się pracy społecznej i będę ot, choćby sadzić... znaczy... marchew".lower()

# Tworzenie zestawu unikalnych znaków
chars = set(data)

# Obliczanie rozmiaru danych i liczby unikalnych znaków
data_size, char_size = len(data), len(chars)

print(f'Rozmiar danych: {data_size}, Liczba unikalnych znaków: {char_size}')


# Mapowanie znaków na indeksy i odwrotnie
char_to_idx = {c:i for i, c in enumerate(chars)}
idx_to_char = {i:c for i, c in enumerate(chars)}

train_X, train_y = data[:-1], data[1:]

# Initialize Network
hidden_size = 25

input_size = char_size + hidden_size
output_size = char_size

lstm = LSTM(input_size, hidden_size, output_size , num_epochs = 1_000, learning_rate = 0.05)

print(input_size)
print(hidden_size)
print(output_size)

