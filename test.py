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

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018',
            'March 3 2001', 'March 3rd 2001', '1 March 2001']

total = len(EXAMPLES)
count = 1

TARGETS = ['1979-05-03', '2009-04-05', '2016-08-21', '2007-07-10', '2018-05-09',
           '2001-03-03', '2001-03-03', '2001-03-01']

for example, target in zip(EXAMPLES, TARGETS):
    source = np.array([string_to_int(example, Tx, human_vocab)])
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_vocab[int(i)] for i in prediction]

    output_date = ''.join(output)
    print("source:", example)
    print("output:", output_date)
    if target == output_date:
        count = count + 1

print('Accuracy({} / {}):{}'.format(count, total, count / total))
