# chainer.nn

## Requirements

- Chainer 2.x
- Python 2.x / 3.x

# Usage

## Basic

### #1

```
import nn

model = nn.Module(
	nn.Linear(None, 1024),
	nn.ReLU(),
	nn.BatchNormalization(1024),
	nn.Linear(None, 512),
	nn.ReLU(),
	nn.BatchNormalization(512),
	nn.Linear(None, 256),
	nn.ReLU(),
	nn.BatchNormalization(256),
	nn.Linear(None, 128),
	nn.ReLU(),
	nn.BatchNormalization(128),
	nn.Linear(None, 10),
)

y = model(x)
```

### #2

```
import nn

model = nn.Module()
model.add(
	nn.Linear(None, 1024),
	nn.ReLU(),
	nn.BatchNormalization(1024),
)
model.add(
	nn.Linear(None, 512),
	nn.ReLU(),
	nn.BatchNormalization(512),
)
model.add(
	nn.Linear(None, 256),
	nn.ReLU(),
	nn.BatchNormalization(256),
)
model.add(
	nn.Linear(None, 128),
	nn.ReLU(),
	nn.BatchNormalization(128),
)
model.add(
	nn.Linear(None, 10),
)

y = model(x)
```

### #3

```
import nn

model = nn.Module()
model.add(
	nn.Linear(None, 1024),
	nn.ReLU(),
	nn.BatchNormalization(1024),
	nn.Linear(None, 512),
	nn.ReLU(),
	nn.BatchNormalization(512),
)
if False:
	model.add(
		nn.Linear(None, 256),
		nn.ReLU(),
		nn.BatchNormalization(256),
		nn.Linear(None, 128),
		nn.ReLU(),
		nn.BatchNormalization(128),
	)
model.add(
	nn.Linear(None, 10),
)

y = model(x)
```

## ResNet

```
import nn

model = nn.Module(
	nn.Residual(
		nn.Convolution2D(None, 64),
		nn.BatchNormalization(),
		nn.ReLU(),
		nn.Convolution2D(None, 64),
		nn.BatchNormalization(),
	),
	nn.ReLU(),
	nn.Residual(
		nn.Convolution2D(None, 64),
		nn.BatchNormalization(),
		nn.ReLU(),
		nn.Convolution2D(None, 64),
		nn.BatchNormalization(),
	),
	nn.ReLU(),
	nn.Residual(
		nn.Convolution2D(None, 64),
		nn.BatchNormalization(),
		nn.ReLU(),
		nn.Convolution2D(None, 64),
		nn.BatchNormalization(),
	),
	nn.ReLU(),
)

y = model(x)
```

## Lambda

```
import nn

model = nn.Module(
	nn.Linear(None, 1024),
	nn.ReLU(),
	nn.BatchNormalization(1024),
	lambda x: x[:, 512:],
	nn.Linear(None, 256),
	nn.ReLU(),
	nn.BatchNormalization(256),
	lambda x: x[:, 128:],
	nn.Linear(None, 64),
	nn.ReLU(),
	nn.BatchNormalization(64),
	lambda x: x[:, 32:],
	nn.Linear(None, 10),
)

y = model(x)
```

## Block

```
import nn

module = nn.Module()
module.add(
	nn.BatchNormalization(1000),
	nn.Linear(1000, 1000),
	nn.ReLU(),
	nn.Dropout(),
)
module.add(
	nn.BatchNormalization(1000),
	nn.Linear(1000, 1000),
	nn.ReLU(),
	nn.Dropout(),
)
module.add(
	nn.BatchNormalization(1000),
	nn.Linear(1000, 1000),
	nn.ReLU(),
	nn.Dropout(),
)

use_batchnorm = True
use_dropout = True

x = np.random.normal(0, 1, (100, 1000)).astype(np.float32)

for block in module.blocks():
	batchnorm, linear, f, dropout = block
	if use_batchnorm:
		x = batchnorm(x)
	x = linear(x)
	x = f(x)
	if use_dropout:
		x = dropout(x)
```

## Submodule

```
import nn

class AutoEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Module(
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 2),
		)
		self.decoder = nn.Module(
			nn.Linear(2, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
		)

autoencoder = AutoEncoder()
x = np.random.normal(0, 1, (100, 1000)).astype(np.float32)
z = autoencoder.encoder(x)
_x = autoencoder.decoder(z)
```

## Serialization

```
import nn

class AutoEncoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = nn.Module(
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 2),
		)
		self.decoder = nn.Module(
			nn.Linear(2, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
			nn.ReLU(),
			nn.Linear(1000, 1000),
		)

autoencoder = AutoEncoder()
autoencoder.save("autoencoder.model")
```
