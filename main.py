"""
This script performs data preprocessing, sequence creation, encoding, train-test split, model building, model training, and testing.

The steps involved are as follows:
1. Load the data from a CSV file.
2. Perform data preprocessing by converting the 'access_date' column to datetime format, removing duplicate rows, and sorting the data by 'user_name' and 'access_date'.
3. Create sequences of app usage for each user.
4. Encode the data using one-hot encoding for the 'app_path' column and convert 'access_date' to the number of seconds since the Unix epoch.
5. Split the data into a training set and a test set.
6. Build a neural network model using PyTorch.
7. Train the model using the training set.
8. Test the model by predicting the next app path given an input of user and app path.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim


# Load the data
data = pd.read_csv('kardia-obfuserdata.csv', header=0)

'''Data Preprocessing'''
# Convert 'access_date' to datetime format
data['access_date'] = pd.to_datetime(data['access_date'])

# Remove rows that are complete duplicates in every value
data = data.drop_duplicates()

# Sort the data by 'user_name' and 'access_date'
data = data.sort_values(by=['user_name', 'access_date'])


'''Sequence Creation'''
# Define a function to create sequences
def create_sequences(df, user_col, item_col, time_col):
    # Add a new column 'day' by extracting the date from 'access_date'
    df['day'] = df[time_col].dt.date
    df = df.sort_values(by=[user_col, 'day'])
    sequences = df.groupby([user_col, 'day'])[item_col].apply(list)
    return sequences

# Create sequences of app usage for each user
sequences = create_sequences(data, "user_name", "app_path", "access_date")

# Save the sequences to a CSV file
sequences.to_csv('sequences.csv')

# '''Encoding'''
# # Create a OneHotEncoder object
# encoder = OneHotEncoder(sparse_output=False)

# # Fit the encoder and transform the 'app_path' column
# onehot = encoder.fit_transform(data["app_path"].values.reshape(-1, 1))

# # Convert the result to a DataFrame
# onehot_df = pd.DataFrame(onehot, columns=encoder.categories_[0])

# # Concatenate the one-hot encoded DataFrame with the original DataFrame
# data_encoded = pd.concat([data, onehot_df], axis=1)

# # Convert 'access_date' to the number of seconds since the Unix epoch
# # print(data_encoded["access_date"])
# data_encoded["access_date"] = data_encoded["access_date"].astype(int) / 10**9

# # One-hot encode 'user_name'
# onehot_user = pd.get_dummies(data_encoded["user_name"])
# data_encoded = pd.concat([data_encoded, onehot_user], axis=1)
# data_encoded = data_encoded.drop("user_name", axis=1)

# # print(data_encoded.head())

# '''Train-Test Split'''
# # Define your features and target variable
# X = data_encoded.drop("app_path", axis=1)
# y = data_encoded["app_path"]

# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Put xtrain data into a csv file
# # X_train.to_csv('X_train.csv', index=False)
# # print("X_train")
# # print(X_train.head())
# # print("y_train")
# # print(y_train.head())


# '''Model Building'''
# # Define the model
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(X_train.shape[1], 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, len(y.unique()))

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# model = Net()

# # Define the loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# '''Model Training'''

# # Initialize a label encoder
# label_encoder = LabelEncoder()

# # Fit the label encoder and transform the target variable
# y_train_encoded = label_encoder.fit_transform(y_train)

# # Convert pandas dataframes to PyTorch tensors
# X_train_tensor = torch.tensor(X_train.values.astype("float32"))
# y_train_tensor = torch.tensor(y_train_encoded)

# # Number of epochs
# epochs = 10

# # Train the model
# for epoch in range(epochs):
#     # Forward pass
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)

#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# # '''Model Evaluation'''
# # # Transform the test set
# # y_test_encoded = label_encoder.fit_transform(y_test)
# # X_test_tensor = torch.tensor(X_test.values.astype("float32"))
# # y_test_tensor = torch.tensor(y_test_encoded)

# # # # Get the model's predictions
# # _, predicted = torch.max(model(X_test_tensor), 1)

# # # # Calculate the accuracy of the model
# # accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)

# # print(f"Accuracy: {accuracy}")


# '''Testing the Mdodel'''
# # Allow me to give an app path and the user, and the model will predict the next app path

# # Define the input
# input = pd.read_csv('test.csv', header=0)

# input["access_date"] = pd.to_datetime(input["access_date"])

# # Create sequences of app usage for each user
# sequences = create_sequences(input, "user_name", "app_path", "access_date")


# """Encoding"""
# # Create a OneHotEncoder object
# encoder = OneHotEncoder(sparse_output=False)

# # Fit the encoder and transform the 'app_path' column
# onehot = encoder.fit_transform(input["app_path"].values.reshape(-1, 1))

# # Convert the result to a DataFrame
# onehot_df = pd.DataFrame(onehot, columns=encoder.categories_[0])

# # Concatenate the one-hot encoded DataFrame with the original DataFrame
# data_encoded = pd.concat([input, onehot_df], axis=1)

# # Convert 'access_date' to the number of seconds since the Unix epoch
# # print(data_encoded["access_date"])
# data_encoded["access_date"] = data_encoded["access_date"].astype(int) / 10**9

# # One-hot encode 'user_name'
# onehot_user = pd.get_dummies(data_encoded["user_name"])
# data_encoded = pd.concat([data_encoded, onehot_user], axis=1)
# data_encoded = data_encoded.drop("user_name", axis=1)

# print(data_encoded.head())
