import random
import csv
import math
from collections import Counter
from random import shuffle

# Generate a sample dataset with random labels
def generate_dataset(file_path, num_rows):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing header row
        writer.writerow(['st_id', 'label', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6'])

        for i in range(num_rows):
            # Generating a row with st_id, random label, and six random numbers
            label = random.randint(0, 1)  # Assuming binary classification for simplicity
            row = [i + 1, label]
            row.extend(random.randint(0, 100) for _ in range(6))
            
            # Writing the row to the CSV file
            writer.writerow(row)
    
    print(f"Sample dataset with labels created: {file_path}")

# Load dataset from the CSV file
def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            dataset.append([int(x) for x in row])
    return dataset

# Calculate Euclidean distance between two data points
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(2, len(row1)):  # Start from 2 to skip st_id and label
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(training_set, test_row, k):
    distances = []
    for train_row in training_set:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# Make a prediction with neighbors
def predict_classification(training_set, test_row, k):
    neighbors = get_neighbors(training_set, test_row, k)
    output_values = [row[1] for row in neighbors]  # Using the label as the class (second column)
    prediction = Counter(output_values).most_common(1)[0][0]
    return prediction

# Split dataset into training and test sets
def train_test_split(dataset, split_ratio=0.8):
    shuffle(dataset)
    split_index = int(len(dataset) * split_ratio)
    return dataset[:split_index], dataset[split_index:]

# Main function to run k-NN
def knn(filename, k=3):
    # Load and split the dataset
    dataset = load_dataset(filename)
    training_set, test_set = train_test_split(dataset)

    # Predict the class for each test instance
    correct_predictions = 0
    for test_row in test_set:
        prediction = predict_classification(training_set, test_row, k)
        actual = test_row[1]  # Assuming the second column is the actual class label
        print(f'Predicted: {prediction}, Actual: {actual}')
        if prediction == actual:
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / len(test_set) * 100.0
    print(f'Accuracy: {accuracy:.2f}%')

# Example usage
file_path = 'random_data_with_labels.csv'
generate_dataset(file_path, num_rows=15)  # Generate 15 rows of data
knn(file_path, k=3)  # Apply k-NN with k=3
