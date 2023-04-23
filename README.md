# Tensorflow Ranger CNN
This is a Tensorflow Implementation of the ranger layer concept described in various paper:

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


# Examples:

You can find an example of usage [Here](./examples/usage_example.py)