# 📬 Spam Email Classifier — Neural Network from Scratch (NumPy)

This project is a **binary spam classifier** built from scratch using only **NumPy** — no machine learning libraries like TensorFlow or Scikit-learn models. It uses a simple feedforward neural network trained on a real SMS spam dataset.

---

## 🚀 Features

- Built entirely with **NumPy**
- Custom implementation of:
  - Forward and backward propagation
  - Cost function (binary cross-entropy)
  - ReLU and Sigmoid activations
  - Gradient descent
- Uses **TF-IDF** features to vectorize email text
- Classifies whether a message is `spam` or `not spam`

---

## 📊 Sample Output
Step 0 | Cost: 0.6931
Step 100 | Cost: 0.5873
Step 500 | Cost: 0.4372
...
🎯 Test Accuracy: 94.1%

Message: "Congrats! You've won a free ticket"
Spam Probability: 0.9612
✅ This is SPAM
----------------------------------------------------------------
## ▶️ How to Run

Install required libraries:

```bash
pip install numpy pandas scikit-learn
python spam_nn.py

**3. Add a "Dataset" section:**
```markdown
## 📂 Dataset

- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- 5,574 English SMS messages labeled as "spam" or "ham"

