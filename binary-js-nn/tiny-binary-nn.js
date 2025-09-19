function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  return y * (1 - y);
}

export class TinyBinaryNN {
  constructor() {
    this.w11 = Math.random() * 2 - 1;
    this.w12 = Math.random() * 2 - 1;
    this.w21 = Math.random() * 2 - 1;
    this.w22 = Math.random() * 2 - 1;
    this.wo1 = Math.random() * 2 - 1;
    this.wo2 = Math.random() * 2 - 1;

    this.b1 = Math.random() * 2 - 1;
    this.b2 = Math.random() * 2 - 1;
    this.bo = Math.random() * 2 - 1;
  }

  calculate(x1, x2) {
    const h1 = sigmoid(x1 * this.w11 + x2 * this.w12 + this.b1);
    const h2 = sigmoid(x1 * this.w21 + x2 * this.w22 + this.b2);
    const out = sigmoid(h1 * this.wo1 + h2 * this.wo2 + this.bo);

    return {h1, h2, out};
  }

  learn(x1, x2, target) {
    const {h1, h2, out} = this.calculate(x1, x2);
    const error = target - out;
    const d_out = dsigmoid(out) * error;

    // ошибка для скрытого слоя
    const d_h1 = d_out * this.wo1 * dsigmoid(h1);
    const d_h2 = d_out * this.wo2 * dsigmoid(h2);

    this.w11 += x1 * d_h1;
    this.w12 += x2 * d_h1;
    this.w21 += x1 * d_h2;
    this.w22 += x2 * d_h2;
    this.b1 += d_h1;
    this.b2 += d_h2;

    this.wo1 += h1 * d_out;
    this.wo2 += h2 * d_out;
    this.bo += d_out;
  }
}
