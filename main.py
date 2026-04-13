import torch

flag = torch.cuda.is_available()
print(flag)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu) else "cpu")
print(device)
print(torch.cuda.get_device_name(device))
print(torch.rand(3,3).to(device))

# check CUDA Version
cuda_version = torch.version.cuda
print("CUDA Version: ",cuda_version)

# check Cudnn version
cudnn_version = torch.backends.cudnn.version()
print("Cudnn Version: ",cudnn_version)