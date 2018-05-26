import sys
import matplotlib.pyplot as plt

lines = open(sys.argv[-1], 'r').readlines()

val_driver_loss = []
val_total_loss = []
train_driver_loss = []
train_total_loss = []
train_acc = []
val_acc = []

for line in lines:
    if line.startswith("Epoch "):
        train_driver_loss.append(float(line.strip().split(', ')[-1].split(' ')[1]))
        train_total_loss.append(float(line.strip().split(', ')[0].split(' ')[-1]))
    elif line.startswith('LB loss'):
        val_driver_loss.append(float(line.strip().split(', ')[-1].split(' ')[-1]))
    elif line.startswith('LB on validation'):
        val_total_loss.append(float(line.strip().split(' ')[-1]))
    elif line.startswith('Accuracy on train'):
        train_acc.append(float(line.strip().split(' ')[-1]))
    elif line.startswith('Accuracy on val'):
        val_acc.append(float(line.strip().split(' ')[-1]))

def equalize_and_scale(arr, n):
    temp = arr[:n]
    return [(t-min(temp))/(max(temp)-min(temp)) for t in temp]

n = len(val_acc)

val_driver_loss = equalize_and_scale(val_driver_loss, n)
val_total_loss = equalize_and_scale(val_total_loss, n)
train_driver_loss = equalize_and_scale(train_driver_loss, n)
train_total_loss = equalize_and_scale(train_total_loss, n)
train_acc = equalize_and_scale(train_acc, n)
val_acc = equalize_and_scale(val_acc, n)

plt.plot(range(n), val_driver_loss, label='val_driver_loss')
plt.plot(range(n), val_total_loss, label='val_total_loss')
plt.plot(range(n), train_driver_loss, label='train_driver_loss')
plt.plot(range(n), train_total_loss, label='train_total_loss')
plt.plot(range(n), train_acc, label='train_acc')
plt.plot(range(n), val_acc, label='val_acc')
plt.legend()
plt.savefig('png/{}.png'.format(sys.argv[-1]))
