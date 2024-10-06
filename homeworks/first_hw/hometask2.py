import unittest

class Element:
    def __init__(self, data, _children=(), _op=''):
        """
        Initializes an Element instance.

        Parameters:
        data: The value of the element.
        _children: A tuple of child elements for the computational graph.
        _op: The operation that produced this element (e.g., '+', '*').
        """
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Element(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """
        Defines addition for Element instances.

        Parameters:
        other: Another Element to add.

        Returns:
        A new Element representing the sum.
        """
        out = Element(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Defines multiplication for Element instances.

        Parameters:
        other: Another Element to multiply.

        Returns:
        A new Element representing the product.
        """
        out = Element(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def relu(self):
        """
        Applies ReLU activation function to the Element.

        Returns:
        A new Element representing the activated value.
        """
        out = Element(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        """
        Computes the gradients of the element and its predecessors through backpropagation.
        """
        self.grad = 1.0
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        for v in reversed(topo):
            v._backward()


class TestElement(unittest.TestCase):

    def test_addition(self):
        a = Element(2.0)
        b = Element(3.0)
        c = a + b
        c.backward()

        self.assertEqual(c.data, 5.0)
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

    def test_multiplication(self):
        a = Element(2.0)
        b = Element(3.0)
        c = a * b
        c.backward()

        self.assertEqual(c.data, 6.0)
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)

    def test_relu(self):
        a = Element(-1.0)
        b = a.relu()
        b.backward()

        self.assertEqual(b.data, 0.0)
        self.assertEqual(a.grad, 0.0)

        a = Element(2.0)
        b = a.relu()
        b.backward()

        self.assertEqual(b.data, 2.0)
        self.assertEqual(a.grad, 1.0)


if __name__ == "__main__":
    unittest.main()
