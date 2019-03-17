# ClassImbalanceLearning
This is the implementation code for the paper "Trainable Undersampling for Class-Imbalance Learning" published in AAAI2019


## File Description

### utils.py: 
load dataset for given task; define some evaluation functions.

### trainer.py: 
implement the policy; perform model training.

### adaptive_trainer.py: 
similar to trainer.py, but trains the policy on gradually increasing data set.

### synthetic.ipynb: 
generate synthetic data; choose supervised classifier and its corresponding hyper-parameters; get results reported in Table 1 of the paper.

### checkerboard.ipynb: 
choose supervised classifier and its corresponding hyper-parameters; plot classification boundaries on original dataset and the sampled dataset with our proposed method as reported in Figure 1 of the paper.

### page.ipynb: 
choose supervised classifier and its corresponding hyper-parameters on page dataset; apply typical data sampling methods to this dataset and the chosen classifier.

### spam.ipynb: 
similar to page.ipynb but performs on the spam message dataset.

### vehicle.ipynb: 
similar to page.ipynb but performs on the vehicle dataset.

### vehicle.ipynb: 
similar to page.ipynb but performs on the creditcard dataset.
