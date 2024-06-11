import torch
from torchvision import datasets, transforms

# # Step 1: Load MNIST dataset
# transform = transforms.Compose([transforms.ToTensor()])
# train_dataset = datasets.MNIST(root=r'C:\Users\hp\Desktop\data', train=True, download=True, transform=transform)
#
# # Step 2: Convert to tensors
# images, labels = train_dataset.train_data, train_dataset.train_labels
#
# # Step 3: Save as .pt file
# torch.save(images, 'mnist_train_data.pt')
# torch.save(labels, 'mnist_train_labels.pt')

###################################################分割线
checkpoint = torch.load('saved_models/MNIST/alt_exp/alt_0_checkpoint.pth.tar')
model_data = checkpoint['model']

# 检查 'state_dict' 键是否存在于 'model' 键下
if 'state_dict' in model_data:
    print("Checkpoint contains 'state_dict' under 'model' key.")
    # 如果需要，可以进一步操作 state_dict
    # state_dict = model_data['state_dict']
else:
    print("Checkpoint does not contain 'state_dict' under 'model' key.")
print(checkpoint.keys())