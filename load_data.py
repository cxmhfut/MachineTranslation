from nmt_utils import *

m = 10000
dataset, human_vocab, machine_vocab, inv_vocab = load_dataset(m)

print(dataset[:10])

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print(X.shape)
print(Y.shape)
print(Xoh.shape)
print(Yoh.shape)

index = 0

print("Source data:", dataset[index][0])
print("Target data:", dataset[index][1])
print()
print("Source data after pre-processing (indices):", X[index])
print("Target data after pre-processing (indices):", Y[index])
print()
print("Source data after pre-processing (one-hot):", Xoh[index])
print("Source data after pre-processing (one-hot):", Yoh[index])
