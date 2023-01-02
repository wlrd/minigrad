import math

class Value:
    
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(data = self.data + other.data, 
                    _children = (self, other),
                    _op='+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(data = self.data * other.data,
                    _children = (self, other),
                    _op='*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad   
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __truediv__(self, other): # self / other
        return self * other**-1
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int or float powers for now"
        out = Value(data=self.data**other,
                    _children=(self,),
                    _op=f'**{other}')
        
        def _backward():
            self.grad += (other * (self.data)**(other-1)) * out.grad   
        out._backward = _backward
        
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x)+1)
        out = Value(data = t,
                   _children=(self,),
                   _op='tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def exp(self):
        x = self.data
        out = Value(data = math.exp(x),
                   _children=(self,),
                   _op='exp')
        
        def _backward():
            # derivative is e^x but out.data is already e^x so we use that
            self.grad += out.data * out.grad 
        out._backward = _backward
        
        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for n in reversed(topo):
            n._backward()