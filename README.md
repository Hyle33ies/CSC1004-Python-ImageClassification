# CSC1004-python-project

## Task description
The project is about using Convolutional Neural Networks (CNN) to classify handwritten digits.
![minist-img](./imgs/abc.webp "minist")

To achieve it, the code will do the following:
- Load the training config.
- Download the minist dataset (divided into training and testing sets).
- Construct the neural network.
- Update the network parameters with training dataset by minimizing the loss. (Training).
- Test the neural network with the testing dataset. (Testing)
- Plot the results.

## How the Deep Learning is implemented

##### The main function

1. Load the config and random seeds.
2. Run the multiprocessing with different seeds.
3. Plot the results. They are recorded by the .txt file.

##### run_with_seed and run

1. Modify the config to the seed assigned, then run with the updated config.
2. In the run function, first assign the seed and device.
3. The rest of the run part was given. It mainly does the following:
   1. Set DataLoader arguments based on whether CUDA is available
   2. Load and preprocess the MNIST dataset
   3. Initialize the DataLoaders for training and testing, model, optimizer, and learning rate scheduler
   4. Train and test the model for each epoch as well as update the leraning rate scheduler. At the same time records the statistics.
   5. Plot the training and testing performance
   6. Save the results to a file and save the trained model
4. 

