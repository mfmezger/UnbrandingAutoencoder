import time
from DataSet import ImageDataSet
import numpy as np
import torch.nn

# Test for GPU
train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if train_on_gpu else "cpu")
pin_memory = True

batch_size = 2
epoch = 20
num_workers = 4
dataset = ImageDataSet(root_dir="")

lossT = [np.inf]
lossL = [np.inf]

# Parallalising the Model and moving it on the GPU, Initializing of the Weights.
model.to(device)
model.apply(U_Net.init_weights)


train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
# torchsummary.summary(model, input_size=(130, 256, 256))

# Initialize the Metrics
initial_lr = 0.001
opt = torch.optim.Adam(model.parameters(), lr=initial_lr)  # try SGD
# opt = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)

# INITIALIZE Scheduler
MAX_STEP = int(1e10)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)

# Generate Folders for the model to be saved.
model.train()
for i in range(epoch):

    # Reset Metrics
    train_loss = 0.0
    since = time.time()

    # TRAININGS LOOP
    for x, y in train_loader:
        # Put the Images & Groundtruth on the GPU.
        x, y = x.to(device).permute(0, 3, 1, 2).float(), y.to(device)

        # Reseting the Optimiser.
        opt.zero_grad()

        # Forward Pass.
        y_pred = model(x)

        # Selecting the Loss.
        lossT = ce_loss(y, y_pred, weights)

        # Update the loss, Optimizer & Scheduler step.
        train_loss += lossT.item() * x.size(0)
        lossT.backward()
        opt.step()
        scheduler.step(i)
        lr = scheduler.get_lr()
        x_size = lossT.item() * x.size(0)


        time_elapsed = time.time() - since

    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('{:.0f}m'.format(train_loss))


torch.save(model.state_dict(), 'model.pth')
