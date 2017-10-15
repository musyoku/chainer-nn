import chainer
import nn

class Model1(nn.Module):
	def __init__(self):
		super().__init__()
		encoder = nn.Module(
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 2),
		)
		encoder.linear = nn.Linear(10, 1000, nobias=True)
		encoder.decoder = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Linear(10, 1000, nobias=True),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 2),
		)
		self.encoder = encoder

class Model2(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Module(
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 2),
		)
		self.encoder.linear = nn.Linear(10, 1000, nobias=True)
		self.encoder.decoder = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Linear(10, 1000, nobias=True),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 2),
		)

def main():
	model1 = Model1()
	model2 = Model2()	
		
	print("model_1")
	for key in dir(model1):
		value = getattr(model1, key)
		if isinstance(value, chainer.Link):
			print(key, value)
	
	print("model_2")
	for key in dir(model2):
		value = getattr(model2, key)
		if isinstance(value, chainer.Link):
			print(key, value)

if __name__ == "__main__":
	main()