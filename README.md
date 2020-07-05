# WideResNet_MNIST_Adversarial_Training

WideResNet implementation on MNIST dataset. FGSM and PGD adversarial attacks on standard training, PGD adversarial training, and Feature Scattering adversarial training.

# Executing Program

For standard training and PGD adversarial training use 

``` bash run.sh```

It automatically executes main.py with additional arguments like no. of iteration, epsilon value, max iterations for attack and step size in each attack. After training the model it executes FGSM and PGD attacks on it.

For feature scattering based adversarial training use

```bash fs_run.sh``` 

Does the same previous one but implements feature scattering based adversarial training.
