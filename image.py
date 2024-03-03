import torch
from model import Generator
from torchvision.utils import save_image

# 모델 가중치 불러오기 
generator = Generator()
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()


# 새로운 잠재 벡터 생성
latent_vector = torch.randn(64, 100)

# 이미지 생성
with torch.no_grad():
    new_images= generator(latent_vector)

# 생성된 이미지 저장
save_image(new_images, 'new_images.png', nrow=8, normalize=True)