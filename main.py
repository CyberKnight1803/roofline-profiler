import argparse 
from tqdm import tqdm

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader

from torchvision.transforms import transforms as T
from torchvision.datasets import ImageFolder

from torchsummary import summary
from calflops import calculate_flops

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


def train(model, dataloader, epochs, learning_rate, start_itr, stop_itr, device, exp_name):

    # Model in train mode
    model.train() 

    # Optimizer 
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    profiler = torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./tensor-logs/{exp_name}'),
        profile_memory=True,
        with_flops=True,
        with_stack=True,
        activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
        ]
    ) 

    iterations = 0
    for epoch in tqdm(range(epochs)):
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):                
            
            # Start Profiler       
            if iterations == start_itr:
                profiler.start()

            with torch.profiler.record_function("Data Loading"):
                data, target = batch 
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            with torch.profiler.record_function("Forward Pass"):
                output = model(data)
            
            with torch.profiler.record_function("Loss Calculation"):
                loss = criterion(output, target)
            
            with torch.profiler.record_function("Backward Pass"):
                loss.backward()
                optimizer.step()

            if iterations == stop_itr:
                profiler.stop()

            iterations += 1
    
    return profiler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, required=True, help='Set experiment name')
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

    # FLOPs: One Iteration
    input_shape = (256, 3, 160, 160)
    flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)

    # Dataloader
    dataloader = create_dataloader(batch_size=args.batch_size)

    # Train Model
    profiler = train(model, dataloader, args.epochs, args.lr, args.start_itr, args.stop_itr, device, args.exp)

    # Profiler stats
    print(profiler.key_averages().table())
