import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from traning1 import MultiModalModel

model_loaded = MultiModalModel(vocab_size, seq_len)
model_loaded.load_state_dict(torch.load("multi_modal_model.pth"))
model_loaded.eval()

sample_text = torch.randint(0, vocab_size, (1, seq_len))
text_out, image_out = model_loaded(sample_text)
generated_image = image_out.detach().squeeze(0).squeeze(0)

plt.imshow(generated_image, cmap="gray")
plt.title("生成图像示例")
plt.show()  