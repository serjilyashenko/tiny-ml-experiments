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

## Backpropagation Formulas

loss function

$E = \frac{1}{2}(\hat{y} - y)^2$

activation function

$y = \sigma(z)$

linear regression

$z(\omega_1, \omega_2, \omega_3): \quad z = \omega_1 x_1 + \omega_2 x_2 + \omega_3 x_3$

gradient descent

$\omega_{01}' = \omega_{01} + \eta \frac{dE}{d\omega_{01}}$

$\frac{dE}{d\omega_1} = \frac{dE}{dy} \cdot \frac{dy}{dz} \cdot \frac{dz}{d\omega_1}$

$(\hat{y} - y) \cdot y(1 - y) \cdot x_1$

weight adjustments

$\omega_1' = \omega_1 + \eta \cdot (\hat{y} - y) \cdot y(1 - y) \cdot x_1$

$\omega_2' = \omega_2 + \eta \cdot (\hat{y} - y) \cdot y(1 - y) \cdot x_2$

$\omega_3' = \omega_3 + \eta \cdot (\hat{y} - y) \cdot y(1 - y) \cdot x_3$

general form

$\omega_i' = \omega_i + \eta \cdot (\hat{y} - y) \cdot y(1 - y) \cdot x_i \quad i \in \{1, 2, 3\}$

## Notes
* Purely for educational purposes.
