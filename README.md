# vayu.

**lightweight deep learning framework to teach you how to backprop**

[![GitHub stars](https://img.shields.io/github/stars/thevoxium/vayu-ml?style=social)](https://github.com/thevoxium/vayu-ml)


## quick start

```cpp
#include "include/value.h"
#include <iostream>

int main() {
    auto a = make_value(2.0);
    auto b = make_value(-3.0);
    auto c = pow(a, 3) - b;  // c = a³ - b
    auto d = c / b;          // d = c / b  
    auto e = -d;             // e = -d

    e->backward();           // compute gradients

    std::cout << "a: " << *a << std::endl;  // shows data and grad
    std::cout << "b: " << *b << std::endl;
    
    return 0;
}
```


## building

```bash
# Clone the repository
git clone https://github.com/thevoxium/vayu-ml.git
cd vayu-ml

# Compile and run
g++ -std=c++17 -O2 -o grad examples/main.cpp src/*.cpp && ./grad
```

## example output

```
a: Value(data=2, grad=4)
b: Value(data=-3, grad=0.888889)
c: Value(data=11, grad=0.333333)
d: Value(data=-3.66667, grad=-1)
e: Value(data=3.66667, grad=1)
```

## additonal commands to compile


```bash
g++ -std=c++17 -O3 -march=native -DUSE_OPENBLAS -I/opt/homebrew/opt/openblas/include -L/opt/homebrew/opt/openblas/lib examples/test.cpp src/tensor.cpp -lopenblas -o grad && ./grad
```


## contributing

Contributions welcome! Open up a PR please.
