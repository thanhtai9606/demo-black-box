# from ProposedMethod.QueryEfficient.Scratch import Attack
from MOAA.MOAA import Attack
from Cifar10Models import Cifar10Model # Can be changes to ImageNetModels
from LossFunctions import UnTargeted, Targeted
import numpy as np
import argparse
import os
from torchvision import datasets, transforms

def get_images_and_labels():
    # Load CIFAR-10 test dataset
    transform = transforms.ToTensor()
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Lấy 1 số ảnh test và label
    x_test = testset.data[:100].astype(np.float32)  # Chuyển sang float32
    y_test = testset.targets[:100]
    y_target = [(y + 1) % 10 for y in y_test]
    
    return x_test, y_test, y_target

if __name__ == "__main__":
    """
    Non-Targeted
    pc = 0.1
    pm = 0.4
    
    Targeted:
    pc = 0.1
    pm = 0.2
    """
    np.random.seed(0)

    pc = 0.1
    pm = 0.4

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="0 or 1", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--save_directory", type=str, default="./results.npy")
    args = parser.parse_args()

    x_test, y_test, y_target = get_images_and_labels()
    model = Cifar10Model(args.model)

    # Thêm vòng lặp để xử lý ảnh đầu tiên
    i = 0  # hoặc args.start nếu muốn bắt đầu từ một vị trí khác
    
    #loss = Targeted(model, y_test[i], y_target[i], to_pytorch=True)
    loss = UnTargeted(model, y_test[i], to_pytorch=True)
    params = {
        "x": x_test[i],  # Chỉ xử lý ảnh thứ i
        "eps": 24, # number of changed pixels
        "iterations": 1000 // 2, # model query budget / population size
        "pc": pc, # crossover parameter
        "pm": pm, # mutation parameter
        "pop_size": 2, # population size
        "zero_probability": 0.3,
        "include_dist": True, # Set false to not consider minimizing perturbation size
        "max_dist": 1e-5, # l2 distance from the original image you are willing to end the attack
        "p_size": 2.0, # Perturbation values have {-p_size, p_size, 0}. Change this if you want smaller perturbations.
        "tournament_size": 2, #Number of parents compared to generate new solutions, cannot be larger than the population
        "save_directory": args.save_directory
    }
    attack = Attack(params)
    print(f"Bắt đầu tấn công ảnh thứ {i}")
    print(f"Label thật: {y_test[i]}")
    print(f"Đang thực hiện tấn công...")
    
    result = attack.attack(loss)
    
    print(f"Kết quả tấn công đã được lưu tại: {args.save_directory}")
