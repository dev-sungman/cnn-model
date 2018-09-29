import torch.nn as nn
import torch.nn.functional as F

class SMNet(nn.Module):
	def __init__(self):
		super(SMNet, self).__init__()
		# nn.Conv2d (input C, output C, size)
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
	
	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return F.log_softmax(out)


class bSMNet(nn.Module):
	def __init__(self):
		super(bSMNet, self).__init__()
		# nn.Conv2d (input C, output C, size)
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.conv3 = nn.Conv2d(16, 16, 1)
		self.conv3_bn = nn.BatchNorm2d(16)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc1_bn = nn.BatchNorm1d(120)
		self.fc2 = nn.Linear(120, 84)
		self.fc2_bn = nn.BatchNorm1d(84)
		nn.Dropout(0.5)
		self.fc3 = nn.Linear(84, 10)
	
	def forward(self, x):
		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3_bn(self.conv3(out)))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc1_bn(self.fc1(out)))
		out = F.relu(self.fc2_bn(self.fc2(out)))
		out = self.fc3(out)
		return F.log_softmax(out)

# smnet Network in Network
class nSMNet(nn.Module):
	def __init__(self):
		super(nSMNet, self).__init__()
		self.classifier = nn.Sequential(
			nn.Conv2d(3, 6, 5),
			nn.ReLU(inplace=True),
			nn.Conv2d(6, 16, 5),
			nn.ReLU(inplace=True),
			nn.Conv2d(16, 10, 1),
			nn.AvgPool2d(kernel_size=5, stride=1, padding=0),
			)
	def forward(self, x):
		x = self.classifier(x)
		x = x.view(x.size(0),-1)
		return x
