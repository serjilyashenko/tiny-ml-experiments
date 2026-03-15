function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  return y * (1 - y);
}

function sigmoidGradient(z) {
  return sigmoid(z) * (1 - sigmoid(z));
}

export class TinyBinaryNN {
  constructor() {
    this.w11 = Math.random();
    this.w12 = Math.random();
    this.b1 = Math.random();

    this.w21 = Math.random();
    this.w22 = Math.random();
    this.b2 = Math.random();

    this.wo1 = Math.random();
    this.wo2 = Math.random();
    this.bo = Math.random();
  }

  calculateZ1(x1, x2) {
    return x1 * this.w11 + x2 * this.w12 + this.b1;
  }

  calculateZ2(x1, x2) {
    return x1 * this.w21 + x2 * this.w22 + this.b2;
  }

  calculateZo(h1, h2) {
    return h1 * this.wo1 + h2 * this.wo2 + this.bo;
  }

  calculate(x1, x2) {
    const z1 = this.calculateZ1(x1, x2);
    const z2 = this.calculateZ2(x1, x2);
    const h1 = sigmoid(z1);
    const h2 = sigmoid(z2);
    const zo = this.calculateZo(h1, h2);
    const out = sigmoid(zo);

    return out;
  }

  learn(x1, x2, target) {
    const lr = 1; // learningRate
    const z1 = this.calculateZ1(x1, x2);
    const z2 = this.calculateZ2(x1, x2);
    const h1 = sigmoid(z1);
    const h2 = sigmoid(z2);
    const zo = this.calculateZo(h1, h2);
    const out = sigmoid(zo);
    const error = target - out;

    const temp0 = lr * error * dsigmoid(out); // or lr * error * sigmoidGradient(zo)

    const temp1 = lr * error * dsigmoid(out) * this.wo1 * dsigmoid(h1); // or lr * error * sigmoidGradient(zo) * this.wo1 * sigmoidGradient(z1)

    const temp2 = lr * error * dsigmoid(out) * this.wo2 * dsigmoid(h2); // or lr * error * sigmoidGradient(zo) * this.wo2 * sigmoidGradient(z2)

    this.wo1 += temp0 * h1;
    this.wo2 += temp0 * h2;
    this.bo += temp0;

    this.w11 += temp1 * x1;
    this.w12 += temp1 * x2;
    this.b1 += temp1;

    this.w21 += temp2 * x1;
    this.w22 += temp2 * x2;
    this.b2 += temp2;
  }
}
