import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import Generator, Discriminator
import os
    
# 하이퍼파라미터 설정
batch_size=16
epochs=200
lr=0.0002
latent_dim=100
sample_interval=500


# 데이터 로더 설정 
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = datasets.ImageFolder(root='./image', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 모델 초기화
generator = Generator()
discriminator = Discriminator()

# 손실함수 및 최적화 
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# 학습 
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # 진짜 이미지와 가짜 이미지에 대한 레이블 생성
        real = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)
        
        # ---------------------
        #  판별자 학습
        # ---------------------
        optimizer_D.zero_grad()

        # 진짜 이미지로 손실 계산
        real_loss = criterion(discriminator(imgs), real)
        
        # 가짜 이미지로 손실 계산
        z = torch.randn(imgs.size(0), latent_dim)
        fake_imgs = generator(z)
        fake_loss = criterion(discriminator(fake_imgs), fake)
        
        # 판별자 손실
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # ---------------------
        #  생성자 학습
        # ---------------------
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim)
        fake_imgs = generator(z)
        g_loss = criterion(discriminator(fake_imgs), real)
        
        g_loss.backward()
        optimizer_G.step()

        
        if i % 100 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            save_image(fake_imgs.data[:25], f"images/{batches_done}.png", nrow=5, normalize=True)

# 학습된 모델 저장
torch.save(generator.state_dict(), 'generator.pth')