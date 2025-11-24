import torch
import torch.nn as nn
import torch.optim as optim
import os

# --- Configuration ---
MODEL_PATH = "simple_model.pth"
NUM_EPOCHS = 5
LEARNING_RATE = 0.01

# 1. Device Setup: Check for CUDA and select the device
# This is the most crucial part for confirming GPU access
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"--- Running on device: {device} ---")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 2. Define a Simple Model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Simple linear layer for testing
        self.fc = nn.Linear(in_features=10, out_features=1) 

    def forward(self, x):
        return self.fc(x)

# 3. Dummy Data Generation
# We create a simple regression task: output is the sum of inputs
# (100 samples, 10 features each)
X_train = torch.randn(100, 10).to(device)
y_train = X_train.sum(dim=1, keepdim=True).to(device)

# 4. Training the Model
print("\n--- Starting Training ---")
model = SimpleNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    # Set model to training mode
    model.train() 
    
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# --- Saving the Model ---
print(f"\n--- Saving model to {MODEL_PATH} ---")
# Recommended way: save the state_dict (learned parameters)
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved successfully.")

# --- Loading and Testing the Model ---
print("\n--- Loading and Testing Saved Model ---")

# 5. Load the Model
# You need to initialize the model architecture first
loaded_model = SimpleNet().to(device)
loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
print("Model loaded successfully.")

# Set model to evaluation mode (important for production/testing)
loaded_model.eval()

# 6. Test with a New Dummy Input
# Test data is also moved to the device
X_test = torch.randn(1, 10).to(device)
y_expected = X_test.sum(dim=1, keepdim=True).item()

# Perform inference (no gradient calculation needed)
with torch.no_grad():
    y_predicted = loaded_model(X_test)
    
y_predicted_item = y_predicted.item()
test_loss = criterion(y_predicted, torch.tensor([[y_expected]]).to(device)).item()

print(f"\nTest Input: {X_test.cpu().numpy()}")
print(f"Expected Output (Sum of Input): {y_expected:.4f}")
print(f"Predicted Output: {y_predicted_item:.4f}")
print(f"Test Loss (MSE): {test_loss:.4e}")

# Check if prediction is reasonably close to expected
if test_loss < 0.1:
    print("\n✅ Test Passed: The prediction is close to the expected output.")
else:
    print("\n❌ Test Failed: Prediction is not close enough. Check training loss.")

# Cleanup (optional)
os.remove(MODEL_PATH)
print(f"Cleaned up: Removed {MODEL_PATH}")