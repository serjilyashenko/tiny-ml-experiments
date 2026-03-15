import { TinyBinaryNN } from './tiny-binary-nn.js'

// XOR
const xorDataSet = [
  [0, 0, 0],
  [0, 1, 1],
  [1, 0, 1],
  [1, 1, 0],
];
// AND
const andDataSet = [
  [0, 0, 0],
  [0, 1, 0],
  [1, 0, 0],
  [1, 1, 1],
];
// OR
const orDataSet = [
  [0, 0, 0],
  [0, 1, 1],
  [1, 0, 1],
  [1, 1, 1],
];
// Choose a dataset
const data = xorDataSet

const nn = new TinyBinaryNN();

for (let i = 0; i < 100_000; i++) {
  const [x1, x2, y] = data[Math.floor(Math.random() * data.length)];
  nn.learn(x1, x2, y);
}

console.log('>>> weights', JSON.stringify({
  w11: nn.w11,
  w12: nn.w12,
  w21: nn.w21,
  w22: nn.w22,
  wo1: nn.wo1,
  wo2: nn.wo2,
}, null, 2));

console.log('[0, 0, 0] -> 0', nn.calculate(0, 0));
console.log('[0, 1, 1] -> 1', nn.calculate(0, 1));
console.log('[1, 0, 1] -> 1', nn.calculate(1, 0));
console.log('[1, 1, 0] -> 0', nn.calculate(1, 1));
