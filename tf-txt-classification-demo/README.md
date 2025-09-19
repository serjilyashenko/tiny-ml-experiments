# TensorFlow Text Classification Mini Demo 📝🤖

This tiny project demonstrates how to use [TensorFlow.js](https://www.tensorflow.org/js) and the pre-trained [toxicity model](https://github.com/tensorflow/tfjs-models/tree/master/toxicity) to classify text as **toxic** or **non-toxic**.

It comes in two flavors:
- **Browser demo** — simple `index.html` file, no setup needed.
- **Node.js CLI demo** — interactive terminal prompt.

---

## 1. Browser Demo

Open `index.html` in your browser and type something into the text box.

- ✅ *Text seems fine* → if no toxic categories detected.
- ⚠️ *Text looks toxic* → if text crosses the confidence threshold (default 0.7).

### Run

Just open `index.html` file in any browser.

---

## 2. Node.js CLI Demo

### Run

```bash
npm install
node index.js
```

