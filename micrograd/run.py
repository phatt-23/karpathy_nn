#!/bin/python

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint 
from graphviz import Digraph

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        # return f"Value(label='{self.label}', data={self.data})"
        return f'Value(data={self.data})'

    def zero_grad(self):
        visited = set()
        def reset(v: Value):
            if v not in visited:
                v.grad = 0.0  # Reset gradient
                visited.add(v)
                for child in v._prev:
                    reset(child)
        reset(self)
    
    def backward(self):
        def topological_sort(root):
            topo = []
            visited = set()
            def build_topo(v: Value):
                if v not in visited:
                    visited.add(v) # pico tohle chybelo
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(root)
            return topo

        self.grad = 1.0     # must first set the outputs gradient to 1.0
        for v in reversed(topological_sort(self)):
            v._backward()

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return self * -1
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Value object only supports powers of integers or floats."

        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        r = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(r, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - r**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

def trace(root: Value):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root: Value | list[Value], filename='expr_graph'):
    roots = root if isinstance(root, list) else [root]

    dot = Digraph(filename=filename, format='pdf', graph_attr={'rankdir': 'LR'}, engine='dot')

    for root in roots:
        nodes, edges = trace(root)
        for n in nodes:
            uid = str(id(n))
            dot.node(name=uid, label='{ %s | data %.4f | grad %.4f}' % (n.label, n.data, n.grad), shape='record')
            if not n._op:
                continue
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

class Neuron:

    def __init__(self, num_in):
        self.w = [Value(random.uniform(-1,1)) for _ in range(num_in)]
        self.b = Value(random.uniform(-1,1))

    def __repr__(self):
        return f'Neuron(w={self.w}, b={self.b})'

    def __call__(self, x):
        #           compute generator expr              accumulator 
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, num_in, num_out):
        self.neurons = [Neuron(num_in) for _ in range(num_out)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()] 

class MLP:

    def __init__(self, num_in: int, num_hiddens: list[int]):
        sz = [num_in] + num_hiddens 
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(num_hiddens))]
    
    def __call__(self, x) -> list[Value]: 
        for layer in self.layers:
            x = layer(x) 
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

def main():
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')  
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')
    x1_w1 = x1 * w1;                x1_w1.label = 'x1*w1'
    x2_w2 = x2 * w2;                x2_w2.label = 'x2*w2'
    x1_w1_x2_w2 = x1_w1 + x2_w2;    x1_w1_x2_w2.label = 'x1*w1 + x2*w2'
    n = x1_w1_x2_w2 + b;            n.label = 'n'

    o = n.tanh();                   o.label = 'o'

    # e = (2*n).exp();                e.label = 'e**2n'
    # o = (e - 1)/(e + 1);            o.label = 'o'

    o.zero_grad()
    o.backward()
    draw_dot(o).render()
    
    # input = [2.0, 3.0]
    # neuron = Neuron(2)
    # neuron(input)
    # print(neuron)
    # print(neuron(input))
    
    # input = [2.0, 3.0, -1.0]
    # layer = Layer(2,3)
    # out = layer(input)
    # print(out)
    
    # x = [2.0, 3.0, -1.0]
    # n = MultiLayerPerceptron(3, [4,4,1])
    # o = n(x)
    # o[0].backward()
    # draw_dot(o).render()
    
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [
        [1.0],
        [-1.0],
        [-1.0],
        [1.0],
    ]
    ys_vals = [[Value(y)] if not isinstance(y, list) else [Value(v) for v in y] for y in ys]

    mlp = MLP(3, [16, 16, 1])

    step_size = 0.01
    epochs = 1000

    losses = []
    for e in range(epochs):

        # forward pass
        y_pred = [mlp(x) for x in xs]
        y_losses = [(a - d)**2 for yvs,ypr in zip(ys_vals, y_pred) for d,a in zip(yvs,ypr)]
        loss = sum(y_losses, start=Value(0));

        losses.append(loss.data)

        # print state
        print('e:', e, 'loss:', loss.data)

        # backward pass
        loss.zero_grad()
        loss.backward()

        # update
        for p in mlp.parameters():
            p.data += p.grad * (-step_size)
        
        if e == epochs - 1:
            draw_dot(y_pred[0][0]).render()

    plt.plot(np.arange(0, epochs, 1), losses, linewidth=2.0)
    plt.show()


if __name__ == '__main__': 
    main()

