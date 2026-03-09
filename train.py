# Import required libraries
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("Starting Diabetes DNN training...")

# Load dataset
df = pd.read_csv("NBD.csv")

# Split features and target
x = df.drop('diabetes', axis=1)
y = df['diabetes']

print("Dataset loaded successfully")
print("Shape:", df.shape)

# Build the model
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Model compiled")

# Train the model
model.fit(x, y, epochs=100, verbose=1)

print("Training completed")

# Test prediction
X_marks = np.array([[45,63]])
prediction = model.predict(X_marks)

print("Prediction for input [45,63]:", prediction)