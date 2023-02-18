import json
import matplotlib.pyplot as plt

# Load data from json file
with open("training_history.json", "r") as f:
    data = json.load(f)

# Extract training data
epochs = list(range(len(data["accuracy"])))
acc = data["accuracy"]
loss = data["loss"]
lr = data["lr"]

# Plot training variables vs epochs
fig, axs = plt.subplots(2, 3, figsize=(15, 8))

axs[0, 0].plot(epochs, acc)
axs[0, 0].set_title("Training Accuracy")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Accuracy")

axs[1, 0].plot(epochs, loss)
axs[1, 0].set_title("Training Total Loss")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Loss")

axs[1, 1].plot(epochs, lr)
axs[1, 1].set_title("Learning Rate")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Learning Rate")

plt.tight_layout()
plt.show()
