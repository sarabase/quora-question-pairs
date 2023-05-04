# Quora Question Pair Challenge Solution

This repository contains the solution to the [Quora Question Pair Challenge](https://www.kaggle.com/c/quora-question-pairs), a natural language processing task that consists of predicting whether a pair of questions asked on Quora are duplicates or not.

## Collaborators

The solution was developed by the following collaborators:

- [@sergibech](https://github.com/sergibech)
- [@socalesc](https://github.com/socalesc)
- [@davidrosado4](https://github.com/davidrosado4)
- [@Goodjorx](https://github.com/Goodjorx)
- [@sarabase](https://github.com/sarabase)

## Repository Structure

The repository is organized as follows:

- `train_models.ipynb`: a Jupyter notebook that contains the code to preprocess the data and train the models. When executed, it creates a folder called `model_artifacts` that contains all the necessary information to reproduce the results.
- `reproduce_results.ipynb`: a Jupyter notebook that contains the code to reproduce the results obtained by our models. It reads from the `model_artifacts` folder.
- `utils.py`: a Python module that contains all the helper functions for the two previous notebooks.

## Reproducing the Results

To reproduce the results obtained by our models, follow these steps:

1. Clone this repository:

```
git clone https://github.com/sarabase/quora-question-pairs.git
```

2. Create a conda environment and install the necessary requirements. Activate the environment:

```
conda create --name quora_test_env --file requirements.txt
conda activate quora_test_env
```

3. Run the train_models.ipynb notebook.
4. Open the reproduce_results.ipynb notebook in Jupyter and execute the cells.


