from keras.optimizers import Adam
from nmt_utils import *
from model import model

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
# model.summary()
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))
model.fit([Xoh,s0,c0],outputs,epochs=10,batch_size=batch_size)
model.save_weights('models/model.h5')
