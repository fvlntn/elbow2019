import core, utils
from torchvision import transforms
import matplotlib.pyplot as plt


# Define custom transforms to apply to your dataset
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(800),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

# Pass in a CSV file instead of XML files for faster Dataset initialization speeds
dataset = core.Dataset('dataset/', transform=custom_transforms)
val_dataset = core.Dataset('dataset_val/')
loader = core.DataLoader(dataset, batch_size=8, shuffle=True) 

model = core.Model(['Elbow'])
losses = model.fit(loader, val_dataset, epochs=10, verbose=True)

plt.plot(losses)  # Visualize loss throughout training
plt.show()

model.save('model_weights.pth')  # Save model to a file

# Directly access underlying torchvision model for even more control
torch_model = model.get_internal_model()
print(type(torch_model))