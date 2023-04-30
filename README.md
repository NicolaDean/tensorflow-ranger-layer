# Tensorflow Ranger CNN
:warning: **[WORK IN PROGRESS]** :warning:\
This is a Tensorflow Implementation of the ranger layer concept described in various paper:
[Ranger Paper](https://arxiv.org/pdf/2003.13874.pdf)
# TODO LIST of Features:
- [ ] Add __init__.py to make ranger easily includable
- [x] Make Ranger Layer modes compatible with tesnorflow op graph (to make them work outside of eager mode)
- [x] Automatically generate Fault Injection Campaign csv report
- [x] Automatically add Ranger after All Convolution and Maxpool from existing models as input
- [x] Helper Function To set Ranger Layers mode in an easy/fast way
- [ ] Make easier to create a model with both Fault Injection Points and Ranger
- [ ] Create an Helper class for Fault Aware Training

# TODO LIST of Examples:
- [ ] Move DataLoading functions and Training functions inside the models itself.

# Principle IDEA:
The scope of this layer is to imporve reliability/robustness of CNN layers against Fault Tollerance of GPU.
To do so, it introduce a domain range for each layer (here is the name Ranger) and for each inference it will handle in adeguate way the outsider values.
3 Possible way to handle Outsiders:
1. Clip them to Zero
2. Threshold them to the Maximum value of the domain
3. Clip them to the mean of the layer values.
   
# Paper of reference:
#TODO

# RANGER helper class features:
1. Allow a plug and play approach to ranger by simply convert existing models in 100% automatic way.
2. Allow to easily Disable, Put in Train Mode, Put in Inference mode all the Ranger layers automatically.
3. Allow to set up the threshold "Strongness" parameter on all the layers.

# Prerequisites:
1. Tensorflow >= 2.0
2. Clone Classes GPU Fault Injector Framework "initialize.sh script" or download it from its repository [github](https://github.com/D4De/classes/tree/dev)


# Examples:

You can find an example of usage [Here](./examples/usage_example.py)
