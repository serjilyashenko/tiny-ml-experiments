/*
  | i1 | i2 | i3 | out |
  |----|----|----|-----|
  | 0  | 0  | 1  | 0   |
  | 1  | 1  | 1  | 1   |
  | 1  | 0  | 1  | 1   |
  | 0  | 1  | 1  | 0   |
  |----|----|----|-----|
  | 1  | 0  | 0  | ?   |
*/

let w1 = Math.random();
let w2 = Math.random();
let w3 = Math.random();

// Neuron input
function neuronInput(i1, i2, i3) {
  return i1 * w1 + i2 * w2 + i3 * w3;
}

// Activation function: Sigmoid
// y = 1/(1+e^-x)
function sigmoid(x) {
  return 1 / (1 + Math.E ** -x);
}

// Neuron output
function out(i1, i2, i3) {
  return sigmoid(neuronInput(i1, i2, i3));
}

// Training

// Sigmoid gradient (derivative)
function sigmoidGradient(x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

// Error cost function (loss function)
// Mean Squared Error
// The division by 2 is not done for meaning, but for convenience when taking the derivative.
function lossFunction(prediction, correct) {
  return (prediction - correct) ** 2 / 2;
}

// Error cost function gradient (derivative)
function lossFunctionGradient(prediction, correct) {
  return correct - prediction;
}

// Adjust weight
function adjustWeight(input, errorInOutput, prediction) {
  return input * errorInOutput * sigmoidGradient(prediction);
}

function teach(i1, i2, i3, correct) {
  const prediction = out(i1, i2, i3);
  const errorInOutput = correct - prediction; // = error cost function gradient
  w1 += adjustWeight(i1, errorInOutput, prediction);
  w2 += adjustWeight(i2, errorInOutput, prediction);
  w3 += adjustWeight(i3, errorInOutput, prediction);
}

for (let i = 0; i <= 100; i++) {
  teach(0, 0, 1, 0);
  teach(1, 1, 1, 1);
  teach(1, 0, 1, 1);
  teach(0, 1, 1, 0);
}

// training data passes
console.log(">> 0 ->", out(0, 0, 1));
console.log(">> 1 ->", out(1, 1, 1));
console.log(">> 1 ->", out(1, 0, 1));
console.log(">> 0 ->", out(0, 1, 1));

// predicted output
console.log(">> 1 ->", out(1, 0, 0));
