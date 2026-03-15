# 🤖VanillaJS Binary Neural Network

This is a tiny neural network implemented from scratch in Vanilla JavaScript that learns the XOR function.
It can be trained to calculate AND or OR operations as well.

It has:
- 2 input neurons
- 2 hidden neurons
- 1 output neuron
- Sigmoid activation + manual backpropagation

## Backpropagation Formulas

### Out neuron

loss function

$E(y): \quad E = \frac{1}{2}(\hat{y} - y)^2$

activation function

$y(z_0): \quad y = \sigma(z_0)$

linear regression

$z_0(\omega_{01}, \omega_{02}): \quad z_0 = \omega_{01} h_1 + \omega_{02} h_2$

gradient descent

$\omega_{01}' = \omega_{01} + \eta \frac{dE}{d\omega_{01}}$

$\frac{dE}{d\omega_{01}} = \frac{dE}{dy} \cdot \frac{dy}{dz_0} \cdot \frac{dz_0}{d\omega_{01}}$

$= (\hat{y} - y) \cdot y(1 - y) \cdot h_1$

weight adjustments

$\omega_{01}' = \omega_{01} + \eta \cdot (\hat{y} - y) \cdot y(1 - y) \cdot h_1$

$\omega_{02}' = \omega_{02} + \eta \cdot (\hat{y} - y) \cdot y(1 - y) \cdot h_2$

### Hidden neuron

$\omega_{11}' = \omega_{11} + \eta \frac{dE}{d\omega_{11}}$

$\frac{dE}{d\omega_{11}} = \frac{dE}{dy} \cdot \frac{dy}{dz_0} \cdot \frac{dz_0}{d\omega_{11}}$

$\frac{dE}{d\omega_{11}} = \frac{dE}{dy} \cdot \frac{dy}{dz_0} \cdot \frac{dz_0}{dh_1} \cdot \frac{dh_1}{dz_1} \cdot \frac{dz_1}{d\omega_{11}}$

$z_0(h_1, h_2): \quad z_0 = \omega_{01} h_1 + \omega_{02} h_2$

$h_1(z_1): \quad h_1 = \sigma(z_1)$

$z_1(\omega_{11}, \omega_{12}): \quad z_1 = \omega_{11} x_1 + \omega_{12} x_2$

$\frac{dE}{d\omega_{11}} = (\hat{y} - y) \cdot y(1-y) \cdot \omega_{01} \cdot h_1(1-h_1) \cdot x_1$

weight adjustments

$\omega_{11}' = \omega_{11} + \eta \cdot (\hat{y} - y) \cdot y(1-y) \cdot \omega_{01} \cdot h_1(1-h_1) \cdot x_1$

$\omega_{12}' = \omega_{12} + \eta \cdot (\hat{y} - y) \cdot y(1-y) \cdot \omega_{01} \cdot h_1(1-h_1) \cdot x_2$

$\omega_{21}' = \omega_{21} + \eta \cdot (\hat{y} - y) \cdot y(1-y) \cdot \omega_{02} \cdot h_2(1-h_2) \cdot x_1$

$\omega_{22}' = \omega_{22} + \eta \cdot (\hat{y} - y) \cdot y(1-y) \cdot \omega_{02} \cdot h_2(1-h_2) \cdot x_2$

## Run it

From `binary-js-nn` folder run:

```bash
node ./index.test.js 
```

## Notes
* Purely for educational purposes.
