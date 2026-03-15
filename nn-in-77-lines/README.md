# 🧠 A single-neuron NN from Scratch

A single-neuron neural network implemented while following the Udemy course [*Machine Learning: Build a Neural Network in 77 Lines of Code*](https://www.udemy.com/course/machine-learning-build-a-neural-network-in-77-lines-of-code/).

---

## Files

| File                     | Description                                                                                  |
|--------------------------|----------------------------------------------------------------------------------------------|
| `./js/manual_weights.js` | Manually picked weights (`w1=1, w2=0, w3=0`) and verifying with Node.js test runner          |
| `./py/neural_network.py` | Python implementation from the course — a `NeuralNetwork` class with `think()` and `train()` |
| `./js/index.js`          | My JavaScript port. Same algorithm, functional style                                         |

## The problem

Training data set:

```
| i1 | i2 | i3 | out |
|----|----|----|-----|
|  0 |  0 |  1 |  0  |
|  1 |  1 |  1 |  1  |
|  1 |  0 |  1 |  1  |
|  0 |  1 |  1 |  0  |
```

Test data: `[1, 0, 0] → 1`

## Notes
* Purely for educational purposes.
