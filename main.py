import argparse 
from tqdm import tqdm

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader

from torchvision.transforms import transforms as T
from torchvision.datasets import ImageFolder

from torchsummary import summary

from models import SModel, MModel, LModel
from config import (
    DEFAULT_ACCELERATOR,
    DEFAULT_MODEL, 
    ALLOWED_MODELS,
    PATH_DATASET, 
    LEARNING_RATE, 
    BATCH_SIZE, 
    MAX_EPOCHS, 
    START_ITR, 
    STOP_ITR
)

def create_dataloader(data_dir: str = PATH_DATASET, batch_size: int = BATCH_SIZE):
    transform = T.Compose([
        T.Resize((160, 160)),          
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(
        root=f"{data_dir}/train", 
        transform=transform
    )

    return DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=True
    )


def train(model, dataloader, epochs, learning_rate, start_itr, stop_itr, device):

    # Model in train mode
    model.train() 

    # Optimizer 
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    iterations = 0
    for epoch in tqdm(range(epochs)):
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):                
            # Start Profiler
            if iterations >= start_itr and iterations < stop_itr:
                
                with torch.autograd.profiler.profile(
                    enabled=True,
                    with_flops=True,
                    profile_memory=True,
                    use_cuda=True,
                ) as profiler:
                    data, target = batch 
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            else:
                data, target = batch 
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            iterations += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--accelerator", type=str, default=DEFAULT_ACCELERATOR, help="Set accelerator")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Set epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Set batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Set Learning rate")
    parser.add_argument("--start_itr", type=int, default=START_ITR)
    parser.add_argument("--stop_itr", type=int, default=STOP_ITR)

    args = parser.parse_args()

    # device
    device = torch.device(args.accelerator if torch.cuda.is_available() else "cpu")

    if args.model == 'small':
        model = SModel()
    
    elif args.model == 'medium':
        model = MModel()

    elif args.model == 'large':
        model = LModel()
    
    else:
        assert args.model in ALLOWED_MODELS, f"Model: {args.model} not allowed. Choose from {ALLOWED_MODELS}"

    # Change to GPU
    model.to(device)
    summary(
        model=model, 
        input_data=(3, 160, 160),
        device=device,
        col_names=['output_size', 'num_params', 'mult_adds']
    )

    # Dataloader
    dataloader = create_dataloader(batch_size=args.batch_size)

    # Train Model
    train(model, dataloader, args.epochs, args.lr, args.start_itr, args.stop_itr, device)
