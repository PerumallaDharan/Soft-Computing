# Import required libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import VGG16, ResNet50, VGG19
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to categorical
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

# Define a simple CNN (LeNet)
def create_lenet():
    model = models.Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))  # Added pool_size argument
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))  # Added pool_size argument
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


# Optimized AlexNet for CIFAR-10 (Smaller and Faster)
def create_alexnet():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))  # Reduced number of neurons
    model.add(layers.Dense(512, activation='relu'))  # Reduced number of neurons
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Use pre-trained VGG16
def create_vgg16():
    base_model = VGG16(weights=None, include_top=False, input_shape=(32, 32, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Use pre-trained ResNet50
def create_resnet50():
    base_model = ResNet50(weights=None, include_top=False, input_shape=(32, 32, 3))
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Function to compile and train the model
def compile_and_train(model, epochs=10):
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), batch_size=64)
    return history

# List of models to train
models = {
    'LeNet': create_lenet(),
    'AlexNet': create_alexnet(),
    'VGG16': create_vgg16(),
    'ResNet50': create_resnet50()
}

# Train each model and display accuracy
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")
    history = compile_and_train(model)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    results[model_name] = test_acc
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")

# Print final results
print("\nComparison of Test Accuracies:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")
