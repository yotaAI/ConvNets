import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class VGG(nn.Module):
	def __init__(self):
		super().__init__()

		self.chosen_features = ['0','5','10','19','28']
		self.model = models.vgg19(pretrained=True).features[:29]

	def forward(self,x):
		features = []

		for layer_num, layer in enumerate(self.model):
			x = layer(x)

			if str(layer_num) in self.chosen_features:
				features.append(x)

		return features


def load_image(image_name,loader):
	image = Image.open(image_name)
	image = loader(image).unsqueeze(0)
	return image.to(device)


#====Main===
if __name__ =='__main__':
	image_size=356
	loader = transforms.Compose(
		[
			transforms.Resize((image_size,image_size)),
			transforms.ToTensor(),
		])

	original_img = load_image("image.jpg",loader)
	style_img = load_image("style.jpg",loader)

	# generated = torch.randn(original_img.shape,device=device,requires_grad=True)
	generated = original_img.clone().requires_grad_(True).to(device)

	#HyperParameters
	total_steps = 1000
	learning_rate=0.01

	alpha = 1
	beta = 0.1
	opt = "LBFGS" #[Adam , LBFGS]

	model = VGG().eval()
	if opt=='Adam':
		optimizer = optim.Adam([generated],lr=learning_rate)
	else:
		optimizer = optim.LBFGS([generated],lr=learning_rate)

	for step in tqdm(range(total_steps)):


		if opt=='Adam':
			generated_features = model(generated)
			original_image_features = model(original_img)
			style_features = model(style_img)
			style_loss , original_loss = 0,0

			for gen_f,orig_f, style_f in zip(generated_features,original_image_features,style_features):

				batch_size,channel,height,width = gen_f.shape
				#Original-Loss
				original_loss += torch.mean((gen_f - orig_f) ** 2)
				#Style-Loss
				G = gen_f.view(channel,height*width).mm(gen_f.view(channel,height*width).t())
				A = style_f.view(channel,height*width).mm(style_f.view(channel,height * width).t())
				style_loss += torch.mean((G - A)**2)

			total_loss = alpha * original_loss  + beta * style_loss

			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()

		else:
			#LBFGS
			def closure():
				generated_features = model(generated)
				original_image_features = model(original_img)
				style_features = model(style_img)
				style_loss , original_loss = 0,0

				for gen_f,orig_f, style_f in zip(generated_features,original_image_features,style_features):

					batch_size,channel,height,width = gen_f.shape
					#Original-Loss
					original_loss += torch.mean((gen_f - orig_f) ** 2)
					#Style-Loss
					G = gen_f.view(channel,height*width).mm(gen_f.view(channel,height*width).t())
					A = style_f.view(channel,height*width).mm(style_f.view(channel,height * width).t())
					style_loss += torch.mean((G - A)**2)

				total_loss = alpha * original_loss  + beta * style_loss

				optimizer.zero_grad()
				total_loss.backward()

				return total_loss

			optimizer.step(closure)

		if step % 10 ==0 :
			save_image(generated,f'generated_adam{step}.png')


