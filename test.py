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

		decoder = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Linear(1000, 2),
		)
		decoder.linear = nn.Linear(10, 1000, nobias=True)
		decoder.generator = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Linear(1000, 2),
		)

		discriminator = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Linear(1000, 2),
		)
		discriminator.linear = nn.Linear(10, 1000, nobias=True)
		discriminator.generator = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Linear(1000, 2),
		)
		decoder.discriminator = discriminator
		encoder.decoder = decoder
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
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Linear(1000, 2),
		)
		self.encoder.decoder.linear = nn.Linear(10, 1000, nobias=True)
		self.encoder.decoder.generator = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Linear(1000, 2),
		)
		self.encoder.decoder.discriminator = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Linear(1000, 2),
		)
		self.encoder.decoder.discriminator.linear = nn.Linear(10, 1000, nobias=True)
		self.encoder.decoder.discriminator.generator = nn.Module(
			nn.GaussianNoise(std=0.3),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Residual(
				nn.Linear(10, 1000, nobias=True),
				nn.ReLU(),
			),
			nn.Linear(1000, 2),
		)

def main():
	model_1 = Model1()
	model_2 = Model2()	
	
	keys_1 = []
	print("model_1")
	for key in dir(model_1):
		value = getattr(model_1, key)
		if isinstance(value, chainer.Link):
			print(key, value)
			keys_1.append(key)
	
	keys_2 = []
	print("model_2")
	for key in dir(model_2):
		value = getattr(model_2, key)
		if isinstance(value, chainer.Link):
			print(key, value)
			keys_2.append(key)

	assert len(keys_1) == len(keys_2)

if __name__ == "__main__":
	main()