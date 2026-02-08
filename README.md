# minimal_autograd_engine
Building automatic differentiation engine, uses computation graphs during backpropagation to compute gradients automatically, from scratch.

---

Autograd Engine is an automatic differentiation system, responsible for computing gradients during neural network training. It dynamically builds a computational graph during the forward pass, tracking all (defined) operations on scalar values that have `.backward()`. This graph enables efficient gradient computation via the chain rule during the backward pass.

---

## Example Usage

### 1. Basic Arithmetic + Backpropagation

```python
from grad_eng import Value

x = Value(2.0)
y = Value(3.0)

f = x * y + x
f.backward()

print(f)        # Value(data=8.0, grad=1.0)
print(x.grad)   # 4.0
print(y.grad)   # 2.0
```

### 2. Non-Linear function

```py
x = Value(3.0)
f = (x**2 - 4).relu()
f.backward()

print(f.data)   # 5.0
print(x.grad)   # 6.0
```
### 3. Neuron with Tanh Activation

```py
x = Value(1.5)
w = Value(-2.0)
b = Value(0.5)

y = (w * x + b).tanh()
y.backward()

print(y.data)
print(x.grad, w.grad, b.grad)
```

### 4. Tiny Neural Network (End-to-End Backprop)

```py
w1, b1 = Value(0.5), Value(0.0)
w2, b2 = Value(-1.0), Value(0.0)

x = Value(2.0)
target = 1.0

h = (w1 * x + b1).tanh()
y = (w2 * h + b2).tanh()

loss = (y - target) ** 2
loss.backward()

print(loss.data)

```
---
>Implementation of [micragrad by Andrej, karpathy](https://github.com/karpathy/micrograd.git)
