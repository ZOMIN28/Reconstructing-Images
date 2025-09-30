import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from model.IRN import IRN
import torch.optim as optim

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# number of subprocesses to use for data loading
num_workers = 1
# every batch load 16 pictures
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2

# ImageNet dataset
data_transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    	     std=[0.229, 0.224, 0.225])
    ])

train_data =torchvision.datasets.ImageFolder(root='ILSVRC2012/train',download=True,transform=data_transform)
train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=4, shuffle=True,num_workers=1)

val_data = torchvision.datasets.ImageFolder(root='ILSVRC2012/val',download=False,transform=data_transform)
val_data_loader =  torch.utils.data.DataLoader(val_data,batch_size=4, shuffle=True,num_workers=1)   

test_data = torchvision.datasets.ImageFolder(root='ILSVRC2012/test',download=False,transform=data_transform)
test_data_loader =  torch.utils.data.DataLoader(test_data,batch_size=4, shuffle=True,num_workers=1)  

# use the pretrained model in ImageNet to extract the feature
VGG19_model = torchvision.models.vgg19(pretrained=True)
VGG19_model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, num_classes),  # we only neet the 4096-feature extraction net
        )

# the image restruction network
model = IRN(in_planes=3)
# lr=0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

n_epochs = 30
valid_loss_min = np.Inf # track change in validation loss
accuracy = []

for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    total_sample = 0
    right_sample = 0
    
    ###################
    # 训练集的模型 #
    ###################
    model.train() #作用是启用batch normalization和drop out
    for data, target in train_data_loader:
        # clear the gradients of all optimized variables（清除梯度）
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        # (正向传递：通过向模型传递输入来计算预测输出)
        output = model(data).to(device)  #（等价于output = model.forward(data).to(device) ）
        # calculate the batch loss
        extracted_feature_gen = VGG19_model(output)  # 提取reconstaction之后图像的特征
        extracted_feature_target = VGG19_model(data)  # 提取原图像的特征
        diff = extracted_feature_gen - extracted_feature_target
        loss = torch.mean(torch.sum(torch.square(diff),dim=1))
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    ######################    
    # 验证集的模型#
    ######################

    model.eval()  # 验证模型
    for data, target in val_data_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data).to(device)
        # calculate the batch loss
        extracted_feature_gen = VGG19_model(output)  # 提取reconstaction之后图像的特征
        extracted_feature_target = VGG19_model(data)  # 提取原图像的特征
        diff = extracted_feature_gen - extracted_feature_target
        loss = torch.mean(torch.sum(torch.square(diff),dim=1))
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class(将输出概率转换为预测类)
        _, pred = torch.max(output, 1)    
        # compare predictions to true label(将预测与真实标签进行比较)
        correct_tensor = pred.eq(target.data.view_as(pred))
        # correct = np.squeeze(correct_tensor.to(device).numpy())
        total_sample += batch_size
        for i in correct_tensor:
            if i:
                right_sample += 1
    print("Accuracy:",100*right_sample/total_sample,"%")
    accuracy.append(right_sample/total_sample)   

    
    # 计算平均损失
    train_loss = train_loss/len(train_data_loader.sampler)
    valid_loss = valid_loss/len(val_data_loader.sampler)
        
    # 显示训练集与验证集的损失函数 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # 如果验证集损失函数减少，就保存模型。
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        torch.save(model.state_dict(), 'cheekpoint/cheekpoint1.pt')
        valid_loss_min = valid_loss
