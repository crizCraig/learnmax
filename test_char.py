import torch

from mingpt.utils import sample
from play_char import train_dataset, trainer, model

checkpoint = torch.load('/home/c2/src/safeobjective/checkpoints/2021_06-20_13-46.54.657999.ckpt')
model.load_state_dict(checkpoint)

context = "O God, O God!"
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(trainer.device)
y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)
