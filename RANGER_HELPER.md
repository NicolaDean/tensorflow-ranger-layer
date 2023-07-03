# Ranger Helper Class Tutorial:

## What it can do??

- Load Ranger after layer of our choice using a "target_function" to select the target layers.
- Compatible with all kind of models, also one with skip connections/ non linear connections.
- Allow to enable or disable or change mode of all ranger layers at once

## How to use it?

```python
model = [... Code ...]

#Load Model into Ranger Helper
RANGER = RANGER_HELPER(model)

#Add Ranger Layer after each Convolutions or Maxpool or Batchnorm
RANGER.convert_model_v2()

#Extract the new Model containing Ranger
ranger_model = RANGER.get_model()
ranger_model.summary()
    
#TUNE THE LAYERS RANGE DOMAIN
RANGER.tune_model_range(x_train)
```