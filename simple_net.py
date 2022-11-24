import srelu
import tensorflow_datasets as tfds
import torch
import torchmetrics
import torchvision
from torchvision import transforms as T
from torch import nn


train_data = tfds.load('german_credit_numeric', split='train[:75%]', as_supervised=True)
test_data = tfds.load('german_credit_numeric', split='train[75%:]', as_supervised=True)
Xtrain, ytrain = train_data.batch(5000).get_single_element()
Xtrain, ytrain = Xtrain.numpy(), ytrain.numpy()
print(Xtrain.shape)
print(ytrain.shape)
Xtest, ytest = test_data.batch(5000).get_single_element()
Xtest, ytest = Xtest.numpy(), ytest.numpy()
print(Xtest.shape)
print(ytest.shape)
from sklearn.preprocessing import normalize
Xtrain=normalize(Xtrain)
Xtest=normalize(Xtest)

tensor_x = torch.Tensor(Xtrain) 
tensor_y = torch.Tensor(ytrain)

my_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) 
train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=64,shuffle=True)

tensor_x_test=torch.tensor(Xtest)
tensor_y_test=torch.tensor(ytest)

my_dataset_test = torch.utils.data.TensorDataset(tensor_x_test,tensor_y_test) 
test_loader = torch.utils.data.DataLoader(my_dataset_test, batch_size=64,shuffle=False)

def accuracy(net, loader, device):
  acc = torchmetrics.Accuracy().to(device)
  for xb, yb in loader:
      xb, yb = xb.to(device), yb.to(device)
      xb = xb.to(torch.float32)
      ypred = net(xb)
      ypred=torch.sigmoid(ypred)
      for i in range(ypred.shape[0]):
        if ypred[i]>0.5:
          ypred[i]=1
        else:
          ypred[i]=0
      #print(ypred)
      pred=torch.split(ypred,int(ypred.shape[0]/net.srelu1.units))
      t=torch.zeros(yb.shape[0],1).cuda()
      for i in range(net.srelu1.units):
        t+=pred[i]
      pred=t/net.srelu1.units   
      _ = acc(pred, yb.reshape(pred.shape[0],1))
  return acc.compute()




class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.srelu1=srelu.SReLU(3)            #2 kernels for the srelu layer
        self.srelu2=srelu.SReLU(1)
        self.fc1 = nn.Linear(24, 10)
        self.fc2=nn.Linear(10,4)
        self.fc3=nn.Linear(4,1)
       

    def forward(self, x):
        x=self.fc1(x)
        #x=F.relu(x)
        x=self.srelu1(x)
        x=self.fc2(x)
        x=self.srelu2(x)
        return self.fc3(x)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
print(device)
model = model().to(device)

loss = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.1)



for i in range(200):

  model.train()
  for xb, yb in train_loader:
    
    xb, yb = xb.to(device), yb.to(device)

    opt.zero_grad()
    
    ypred=model(xb)

    pred=torch.split(ypred,int(ypred.shape[0]/model.srelu1.units))
    s=torch.zeros(yb.shape[0],1).cuda()
    for j in range(model.srelu1.units):
      s=s+pred[j]
    pred=s/model.srelu1.units                      #take the mean of the predictions by the units kernels of the srelu layer
    l = loss(pred, yb.reshape(pred.shape[0],1))     
    #print(l.item())
    l.backward()
    opt.step()
    

  model.eval()
  
  print(f'Accuracy at epoch {i}: {accuracy(model, test_loader, device)}')
  if(accuracy(model, test_loader, device)>0.77): break
