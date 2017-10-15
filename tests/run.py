import chainer, cupy
import numpy as np
import nn

class Model1(nn.Module):
	def __init__(self):
		super(Model1, self).__init__()
		prime = nn.Module(
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 2),
		)
		prime.onslaught = nn.Linear(10, 1000, nobias=True)

		megatron = nn.Module(
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
		megatron.berserker = nn.Linear(10, 1000, nobias=True)
		megatron.mohawk = nn.Module(
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

		quintessa = nn.Module(
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
		quintessa.nitrozeus = nn.Linear(10, 1000, nobias=True)
		quintessa.dreadbot = nn.Module(
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
		megatron.quintessa = quintessa
		prime.megatron = megatron
		self.prime = prime
		self.cluster_head = nn.Linear(10, 10, nobias=True)

class Model2(nn.Module):
	def __init__(self):
		super(Model2, self).__init__()
		self.prime = nn.Module(
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 2),
		)
		self.prime.onslaught = nn.Linear(10, 1000, nobias=True)
		self.prime.megatron = nn.Module(
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
		self.prime.megatron.berserker = nn.Linear(10, 1000, nobias=True)
		self.prime.megatron.mohawk = nn.Module(
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
		self.prime.megatron.quintessa = nn.Module(
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
		self.prime.megatron.quintessa.nitrozeus = nn.Linear(10, 1000, nobias=True)
		self.prime.megatron.quintessa.dreadbot = nn.Module(
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
		self.cluster_head = nn.Linear(10, 10, nobias=True)

def check_cupy_ndarray(module):
	for key in dir(module):
		value = getattr(module, key)
		if isinstance(value, (chainer.Chain, Model1, Model2)):
			continue
		if isinstance(value, chainer.Link):
			assert isinstance(value.W.data, cupy.core.core.ndarray)

def compare_layers(a, b):
	keys_a = []
	for key in dir(a):
		value = getattr(a, key)
		if isinstance(value, chainer.Chain):
			continue
		if isinstance(value, chainer.Link):
			keys_a.append(key)
	
	keys_b = []
	for key in dir(b):
		value = getattr(b, key)
		if isinstance(value, chainer.Chain):
			continue
		if isinstance(value, chainer.Link):
			keys_b.append(key)

	print("a")
	for key in keys_a:
		print(key)
	print("b")
	for key in keys_b:
		print(key)
	assert len(keys_b) == len(keys_a)

def main():
	model_1 = Model1()
	model_2 = Model2()
	model_1.to_gpu()
	model_2.to_gpu()

	compare_layers(model_1.prime.megatron.quintessa, model_2.prime.megatron.quintessa)
	check_cupy_ndarray(model_1.prime.megatron.quintessa)
	check_cupy_ndarray(model_2.prime.megatron.quintessa)

	compare_layers(model_1.prime.megatron, model_2.prime.megatron)
	check_cupy_ndarray(model_1.prime.megatron)
	check_cupy_ndarray(model_2.prime.megatron)

	compare_layers(model_1.prime, model_2.prime)
	check_cupy_ndarray(model_1.prime)
	check_cupy_ndarray(model_2.prime)

	compare_layers(model_1, model_2)
	check_cupy_ndarray(model_1)
	check_cupy_ndarray(model_2)

	module = nn.Module(
		nn.Linear(1000, 1000),
		nn.ReLU(),
		nn.Linear(1000, 1000),
		nn.ReLU(),
	)
	module.mean = nn.Linear(1000, 2)
	module.ln_var = nn.Linear(1000, 2)

	x = np.random.normal(0, 1, (100, 1000)).astype(np.float32)

	internal = module(x)
	mean = module.mean(internal)
	ln_var = module.ln_var(internal)
	z = chainer.functions.gaussian(mean, ln_var)
	
if __name__ == "__main__":
	main()