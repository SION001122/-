import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 함수 정의 (이미지 데이터와 레이블을 NumPy 배열로 반환)
def load_data(data_dir):
    classs = os.listdir(data_dir)
    images = []
    labels = []
    
    for label, emotion in enumerate(classs): # 감정 폴더 순회
        emotion_dir = os.path.join(data_dir, emotion) # 감정 폴더 경로
        for image_name in os.listdir(emotion_dir): # 이미지 파일 순회
            image_path = os.path.join(emotion_dir, image_name) # 이미지 파일 경로
            image = Image.open(image_path).convert('L')  # 흑백 이미지로 변환
            image = image.resize((48, 48)) # 이미지 크기 조정
            image = np.array(image)
            images.append(image)
            labels.append(label)
            
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    """CNN 블록: 입력 크기 유지"""
    def __init__(self, in_channels=1, out_channels=64):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.block(x)
        return x

class SimpleCnn(nn.Module):
    """3개의 CNN 블록을 쌓은 모델"""
    def __init__(self):
        super(SimpleCnn, self).__init__()
        self.block1 = CNNBlock(in_channels=1, out_channels=64)
        self.block2 = CNNBlock(in_channels=64, out_channels=128)
        self.block3 = CNNBlock(in_channels=128, out_channels=256)
        self.pool = nn.AdaptiveAvgPool2d((6, 6))  # 폴링 레이어
        self.flatten = nn.Flatten() # Flatten 레이어에 입력
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 16),  # 최종 출력 크기는 7
            nn.ReLU(),
            nn.Linear(16, 7)
        )

    def forward(self, x):
        x = self.block1(x)
        x = F.dropout(x, p=0.2 + torch.rand(1).item() * 0.1, training=self.training)  # 0.2 ~ 0.3 사이의 동적 드롭아웃
        #동적 드롭확률 범위 : 0 ~ 0.3 0.2~0.3으로 하려면 
        x = self.block2(x)
        x = F.dropout(x, p=0.1 + torch.rand(1).item() * 0.1, training=self.training)  # 0.1 ~ 0.2 사이의 동적 드롭아웃
        x = self.block3(x) #드롭아웃 적용 안함
        #training=self.training : 학습 중일 때만 드롭아웃 적용
        x = self.pool(x)
        x = self.flatten(x)
        
        # FC 레이어 처리
        for layer in self.fc:
            x = layer(x)
            if isinstance(layer, nn.ReLU):  # 활성화 함수 뒤에서 드롭아웃 적용
               x = F.dropout(x, p=0.2 + torch.rand(1).item() * 0.1, training=self.training)  # 0.2 ~ 0.3 사이의 동적 드롭아웃
        return x
    
model = SimpleCnn()


#DataLoader와 
from torch.utils.data import DataLoader, TensorDataset
#gpu 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Train 데이터 로드
train_data_dir = 'train'  # train 디렉토리 경로
train_images, train_labels = load_data(train_data_dir) # train 데이터 로드
train_loader = (train_images, train_labels)  # train 데이터를 튜플로 묶음
# Test 데이터 로드
test_data_dir = 'test'  # test 디렉토리 경로
test_images, test_labels = load_data(test_data_dir)
test_loader = (test_images, test_labels)  # test 데이터를 튜플로 묶음

# 예시: 첫 번째 축에서 3채널만 유지

#gpu로 데이터 이동
train_images = torch.tensor(train_images, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
test_images = torch.tensor(test_images, dtype=torch.float32).to(device)
test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
train_loader = (train_images, train_labels)
test_loader = (test_images, test_labels)

# 데이터 로드 후 채널 차원 추가
train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1).to(device)
test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1).to(device)

# TensorDataset 생성
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

# DataLoader로 데이터셋 로드
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
print(f"Train shape: {train_images.shape}")  # 텐서의 크기를 출력   


# 손실 함수와 옵티마이저 정의
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 모델 학습
num_epochs = 300
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:
            print(f"회차: {epoch + 1}, 배치: {i + 1}, 손실: {running_loss / 10}")
            running_loss = 0.0
            
print("학습 완료") # 학습 완료 메시지 출력

# 모델 평가
correct = 0
total = 0

# 각 클래스별 정확도 계산 및 차트화
class_correct = list(0. for _ in range(7)) # 클래스별 정답 개수를 저장할 리스트 초기화
class_total = list(0. for _ in range(7)) # 클래스별 전체 개수를 저장할 리스트 초기화
 
with torch.no_grad(): # 평가 과정에서는 기울기를 계산하지 않음
    for data in test_loader: # 테스트 데이터 순회
        images, labels = data # 이미지와 레이블을 가져옴
        outputs = model(images) # 모델로 예측
        _, predicted = torch.max(outputs, 1) # 가장 높은 값의 인덱스를 예측값으로 사용 (소프트맥스 함수 사용)
        c = (predicted == labels).squeeze() # 예측값과 실제값을 비교하여 일치 여부를 구함
        
        for i in range(len(labels)): # 레이블의 개수만큼 반복
            label = labels[i] # 레이블을 가져옴
            class_correct[label] += c[i].item() # 클래스별로 정답 개수를 계산
            class_total[label] += 1 # 클래스별로 전체 개수를 계산

# 레이블 이름
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 클래스별 정확도 출력
for i in range(7):
    print(f"정확도 {emotion_labels[i]}: {100 * class_correct[i] / class_total[i]:.2f}%")

# 차트 시각화
plt.figure(figsize=(10, 5))  # 차트 크기 설정
plt.bar(emotion_labels, [100 * class_correct[i] / class_total[i] for i in range(7)])  # 막대 차트 생성
plt.xlabel('감정')  # x축 레이블 
plt.ylabel('정확도 (%)')  # y축 레이블
plt.title('감정 정확도')  # 차트 제목
plt.show()
