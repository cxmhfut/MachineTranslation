from keras.layers import RepeatVector, Concatenate
from keras.layers import Dense, Activation, Dot
from keras.layers import Input, Bidirectional, LSTM
from keras.models import Model
from keras.utils import plot_model
from nmt_utils import *

Tx = 30
Ty = 10

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation='tanh')
densor2 = Dense(1, activation='relu')
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes=1)


def one_step_attention(a, s_prev):
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])

    return context


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')

    s = s0
    c = c0

    outputs = []

    a = Bidirectional(LSTM(n_a, return_sequences=True), name='bidirectional_1')(X)
    post_activation_LSTM_cell = LSTM(n_s, return_state=True)
    output_layer = Dense(machine_vocab_size, activation='softmax')

    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        out = output_layer(s)
        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model


if __name__ == '__main__':
    Tx = 30
    Ty = 10
    n_a = 32
    n_s = 64
    m = 10000
    dataset, human_vocab, machine_vocab, inv_vocab = load_dataset(m)
    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
    model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
    model.summary()
    plot_model(model, to_file='model.png')
