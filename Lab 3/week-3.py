import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, input_size, learning_rate=1, iterations=10):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.iterations = iterations

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation_function(z)

    def train(self, X, y):
        for _ in range(self.iterations):
            print(f"\nIteration {_+1}:")
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                print(f"Sample: {xi}")
                print(f"Prediction: {prediction}, Target: {target}")
                print(f"Current weights: {self.weights}")
                if prediction != target:
                    self.weights[1:] += self.learning_rate * (target-prediction) * xi
                    self.weights[0] += self.learning_rate * (target )
                    print(f"Updated weights: {self.weights}")
                    print(f"Updated bias: {self.weights[0]}")
                else:
                    print("No Update in weights required")
                print("\n")
    def evaluate(self, X, y):
        predictions = [self.predict(xi) for xi in X]
        accuracy = np.mean(predictions == y)
        return accuracy

df = pd.read_csv('random_data.csv')

X = df[['c1', 'c2', 'c3', 'c4', 'c5', 'c6']].values
y = df['result'].values

perceptron = Perceptron(input_size=X.shape[1], learning_rate=1, iterations=10)

perceptron.train(X, y)
accuracy = perceptron.evaluate(X, y)
print(f"Training accuracy: {accuracy * 100:.2f}%")

new_marks = np.array([7,0,2,3,4,5]) 
prediction = perceptron.predict(new_marks)
print(f"Prediction for new marks (Pass=1, Fail=0): {prediction}")
