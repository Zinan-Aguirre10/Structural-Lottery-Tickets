# Structural-Lottery-Tickets
A group research project for Machine Learning that explored finding and determining if structured winning lottery tickets exist in neural networks through iterative structured pruning. The characteristics of a winning lottery ticket must follow from the The Lottery Ticket Hypothesis paper. 

## Overview:
In the Lottery Ticket Hypothesis paper, researched focused on finding such tickets in neural networks from unstructured pruning by masking weights. This use of unstructured pruning does not produce practical speedup on standard hardware. Thus our project instead explored removing entire neurons and convolutional filters ultimately creating smaller and more efficeint architectures. We took these pruned networks and reset them with their original initialization and compared them to networks that were randomly reinitialized.

## Research Goal:
The goal of this paper is find such structured winning tickets that:
 1. Achieved similar test accuracy in comparison to the base unpruned model.
 2. Are able to be trained efficiently after resetting surviving weights to their original initialization.
 3. Are a superior model over randomly reinitialized pruned networks of the same architecture. 

## Models & Datasets:
### LENET on MNIST
- Fully connected architecture: 784 → 300 → 100 → 10
- Dataset: MNIST handwritten digit classification.
- Iterative structural neuron pruning.

### CONV-2 on CIFAR-10
(Primary area of my contribution)
- Two-layer convolutional neural network.
- Architecture: Conv(64) → Conv(64) → Pool FC(256) → FC(256) → Output(10)
- Dataset: CIFAR-10 image classification.
- Applied iterative layer-wise structural pruning:
    - 5% filter pruning in convolutional layers.
    - 10% neuron pruning in fully connected layers.
- Evaluated training convergence and test accuracy across pruning rounds.

### CONV-4 on CIFAR-10
- Deeper convolutional architecture with four convolutional layers.
- Hybrid pruning strategy combining global and layer-wise pruning methods.

## My Contributions
My primary focus and contribution in this project was the CONV-2 structured pruning experiments on CIFAR-10. Contributions included:
- I Implemented and tested iterative structural pruning procedures for convolutional neural networks.
- Worked on layer-wise filter pruning and neuron pruning strategies.
- Trained and evaluated Winning Ticket vs Random Reinitialization models.
- Analyzed model accuracy, convergence behavior, and pruning efficiency.
- Collaborated with teammates on experiment design, debugging, and result interpretation.

## Key Findings
- We found that structured pruning was able to significantly reduce model size while preserving much of the original test accuracy for a large portion of its reduced size. 
- CONV-2 maintained near-baseline accuracy through many pruning rounds despite major parameter reduction.
- Winning Ticket initialization did not consistently outperform random reinitialization.
- Training difficulty increased substantially as networks became more pruned.
- Results suggest that identifying structured lottery tickets is more difficult than identifying unstructured sparse subnetworks and more complicated then just resetting their weights back to original intialization.
## Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib
- CIFAR-10 / MNIST datasets

## Skills Demonstrated
- Deep Learning
- Convolutional Neural Networks (CNNs)
- Model Compression
- Structured Pruning
- Experimental Design
- Data Analysis & Visualization
- Research Collaboration
- Scientific Writing

## Reference
Frankle & Carbin (2019), The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
