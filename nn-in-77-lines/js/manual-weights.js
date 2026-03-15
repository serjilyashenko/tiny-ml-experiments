const assert = require("node:assert");
const test = require("node:test");

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

test("My manual weights", () => {
  let w1 = 1;
  let w2 = 0;
  let w3 = 0;

  function out(i1, i2, i3) {
    return i1 * w1 + i2 * w2 + i3 * w3;
  }

  // training data passes
  assert.equal(out(0, 0, 1), 0);
  assert.equal(out(1, 1, 1), 1);
  assert.equal(out(1, 0, 1), 1);
  assert.equal(out(0, 1, 1), 0);

  // predicted output
  assert.equal(out(1, 0, 0), 1);
});
