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

### #1

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

### #1

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