import math

class Value:
    """
    A scalar value that supports automatic differentiation.
    Each Value stores:
    - data: the scalar value
    - grad: the gradient of some final output w.r.t this value
    - _prev: parent nodes in the computation graph
    - _op: operation that produced this node
    - _backward: function to propagate gradients to parents
    """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0

        # Internal graph bookkeeping
        self._prev = set(_children)   # parents in the graph
        self._op = _op                # operation name
        self._backward = lambda: None # populated during forward pass

    # --------------------
    # Arithmetic Operators
    # --------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # dout/dself = 1, dout/dother = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(xy)/dx = y, d(xy)/dy = x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Exponent must be int or float"
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # d(x^n)/dx = n*x^(n-1)
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    # --------------------
    # Activation Functions
    # --------------------

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # Gradient flows only if input > 0
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out
    
    def tanh(self):
        """
        Hyperbolic tangent activation.
        """

        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out
    
    def softmax(values):
        """
        Softmax over a list of Value objects.
        Returns a list of Value objects.
        """

        # Numerical stability trick
        max_val = max(v.data for v in values)
        exps = [math.exp(v.data - max_val) for v in values]
        sum_exps = sum(exps)

        out = []
        for i, v in enumerate(values):
            s = exps[i] / sum_exps
            out.append(Value(s, (v,), 'softmax'))

        def backward():
            for i in range(len(values)):
                for j in range(len(values)):
                    if i == j:
                        values[j].grad += out[i].grad * out[i].data * (1 - out[j].data)
                    else:
                        values[j].grad += -out[i].grad * out[i].data * out[j].data

        for o in out:
            o._backward = backward

        return out


    # --------------------
    # Backpropagation
    # --------------------

    def backward(self):
        """
        Performs reverse-mode autodiff.
        Steps:
        1. Topologically sort the computation graph
        2. Seed gradient of output with 1
        3. Traverse graph in reverse order and apply chain rule
        """

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Seed gradient
        self.grad = 1.0

        # Apply chain rule
        for node in reversed(topo):
            node._backward()

    # --------------------
    # Convenience Operators
    # --------------------

    def __neg__(self):          # -x
        return self * -1

    def __sub__(self, other):   # x - y
        return self + (-other)

    def __rsub__(self, other):  # y - x
        return other + (-self)

    def __truediv__(self, other):   # x / y
        return self * other**-1

    def __rtruediv__(self, other):  # y / x
        return other * self**-1

    def __radd__(self, other):  # y + x
        return self + other

    def __rmul__(self, other):  # y * x
        return self * other

    # --------------------
    # Debugging
    # --------------------

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
