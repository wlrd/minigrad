from nn import MLP

n = MLP(3, [4, 4, 1])

# toy example of inputs and targets
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# gradient descent
for k in range(100):

    # forward pass
    ypreds = [n(x) for x in xs]
    loss = sum((y-ypred)**2 for y, ypred in zip(ys, ypreds))

    # backward pass

    # MAKE SURE TO SET ZEROGRAD BEFORE BACKPROP
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update weights
    for p in n.parameters():
        p.data += -.1 * p.grad

    print(k, loss.data)