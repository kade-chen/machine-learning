import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# 1. 简单文本编码器
# -----------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        return h_n.squeeze(0)  # [batch, hidden_dim]

# -----------------------------
# 2. 文本解码器
# -----------------------------
class TextDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_size, seq_len):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, seq_len * vocab_size)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
    def forward(self, h):
        out = self.fc(h)
        return out.view(-1, self.seq_len, self.vocab_size)

# -----------------------------
# 3. 图像解码器
# -----------------------------
class ImageDecoder(nn.Module):
    def __init__(self, hidden_dim, img_size=28*28):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, img_size)
        self.img_size = img_size
        
    def forward(self, h):
        img_vector = self.fc(h)
        return img_vector.view(-1, 1, 28, 28)

# -----------------------------
# 4. 多模态模型
# -----------------------------
class MultiModalModel(nn.Module):
    def __init__(self, vocab_size, seq_len, hidden_dim=64):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, hidden_dim=hidden_dim)
        self.text_decoder = TextDecoder(hidden_dim, vocab_size, seq_len)
        self.image_decoder = ImageDecoder(hidden_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        text_out = self.text_decoder(h)
        image_out = self.image_decoder(h)
        return text_out, image_out

# -----------------------------
# 5. 超参数与数据
# -----------------------------
vocab_size = 100
seq_len = 5
batch_size = 16
num_epochs = 10
lr = 1e-3

def get_batch(batch_size):
    text_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_text = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_image = torch.rand(batch_size, 1, 28, 28)
    return text_tokens, target_text, target_image

# -----------------------------
# 6. 模型、损失、优化器
# -----------------------------
model = MultiModalModel(vocab_size, seq_len)
criterion_text = nn.CrossEntropyLoss()
criterion_image = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------------
# 7. 训练循环
# -----------------------------
for epoch in range(num_epochs):
    text_batch, target_text, target_image = get_batch(batch_size)
    optimizer.zero_grad()
    text_pred, image_pred = model(text_batch)
    
    # 文本 loss
    loss_text = criterion_text(
        text_pred.view(-1, vocab_size), 
        target_text.view(-1)
    )
    # 图像 loss
    loss_image = criterion_image(image_pred, target_image)
    
    loss = loss_text + loss_image
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# -----------------------------
# 8. 保存模型
# -----------------------------
torch.save(model.state_dict(), "multi_modal_model.pth")
print("模型已保存：multi_modal_model.pth")

# -----------------------------
# 9. 加载模型并生成示例
# -----------------------------
model_loaded = MultiModalModel(vocab_size, seq_len)
model_loaded.load_state_dict(torch.load("multi_modal_model.pth"))
model_loaded.eval()

sample_text = torch.randint(0, vocab_size, (1, seq_len))
text_out, image_out = model_loaded(sample_text)
generated_image = image_out.detach().squeeze(0).squeeze(0)

plt.imshow(generated_image, cmap="gray")
plt.title("生成图像示例")
plt.show()