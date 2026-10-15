import torch

a = torch.tensor([1.0, 2.0, 3.0], device="mps")
dot = torch.dot(a, a)

print("Версия PyTorch ", torch.__version__)
print("Доступность CUDA ", torch.cuda.is_available())
print("Доступность GPU через MPS ", torch.backends.mps.is_available())
print("==========================")
print("Какой-то Тензор хз что это ", a.device)
print("Скалярное произведение = ", dot)
print("Число = ", dot.item())