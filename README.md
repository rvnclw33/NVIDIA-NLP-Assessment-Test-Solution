# NVIDIA-NLP-Assessment-Test-Solution

# The Problem
The Federalist Papers - History Mystery!
The Federalist Papers are a set of essays written between 1787 and 1788 by Alexander Hamilton, James Madison and John Jay. Initially published under the pseudonym 'Publius', their intent was to encourage the ratification of the then-new Constitution of the United States of America. In later years, a list emerged where the author of each one of the 85 papers was identified. Nevertheless, for a subset of these papers the author is still in question. The problem of the Federalist Papers authorship attribution has been a subject of much research in statistical NLP in the past. Now you will try to solve this question with your own BERT-based project model.Image

In concrete terms, the problem is identifying, for each one of the disputed papers, whether Alexander Hamilton or James Madison are the authors. For this exercise, you can assume that each paper has a single author, i.e., that no collaboration took place (though that is not 100% certain!), and that each author has a well-defined writing style that is displayed across all the identified papers.

# Your Project
You are provided with labeled train.tsv and dev.tsv datasets for the project. There are 10 test sets, one for each of the disputed papers. All datasets are contained in the data/federalist_papers_HM directory.

Each "sentence" is actually a group of sentences of approximately 256 words. The labels are '0' for HAMILTON and '1' for MADISON. There are more papers by Hamilton in the example files than by Madison. The validation set has been created with approximately the same distribution of the two labels as in the training set.

Your task is to build neural networks using NeMo, as you did in Lab 2. You'll train your model and test it. Then you'll use provided collation code to see what answers your model gives to the "history mystery"!

Along the way, you'll save code snippets that will be tested with the autograder once you are done. Submission instructions are provided at the end of the notebook for this part.

# Resources and Hints
* **Example code:**<br>
In the file browser at your left, you'll find the `lab2_reference_notebooks` directory.  This contains solution notebooks from Lab 2 for text classification and NER to use as examples.
* **Language model (PRETRAINED_MODEL_NAME):**<br>
You may find it useful to try different language models to better discern style.  Specifically, it may be that capitalization is important, which would mean you'd want to try a "cased" model.
* **Maximum sequence length (MAX_SEQ_LEN):**<br>
Values that can be used for MAX_SEQ_LENGTH are 64, 128, or 256.  Larger models (BERT-large, Megatron) may require a smaller MAX_SEQ_LENGTH to avoid an out-of-memory error.
* **Number of Classes (NUM_CLASSES):**<br>
For the Federalist Papers, we are only concerned with HAMILTON and MADISON.  The papers by John Jay have been excluded from the dataset.
* **Batch size (BATCH_SIZE):**<br>
Larger batch sizes train faster, but large language models tend to use up the available memory quickly.
* **Memory usage:**<br>
Some of the models are very large.   If you get "RuntimeError: CUDA out of memory" during training, you'll know you need to reduce the batch size, sequence length, and/or choose a smaller language model, restart the kernel, and try again from the beginning of the notebook.
* **Accuracy and loss:**<br>
It is definitely possible to achieve 95% or more model accuracy for this project.  In addition to changes in accuracy as the model trains, pay attention to the loss value.  You want the loss value to be dropping and getting very small for best results.
* **Number of epochs (NUM_EPOCHS):**<br>
You may need to run more epochs for your model (or not!).
