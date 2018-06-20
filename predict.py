from model import model
from nmt_utils import *

m = 10000
Tx = 30
Ty = 10
n_a = 32
n_s = 64
learning_rate = 0.005
batch_size = 100

dataset, human_vocab, machine_vocab, inv_vocab = load_dataset(m)
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

model.load_weights('models/model_50.h5')

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))

while (True):
    input_data = input('Input data:')
    source = np.array([string_to_int(input_data, Tx, human_vocab)])
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_vocab[int(i)] for i in prediction]
    output_date = ''.join(output)
    print('output data:', output_date)
