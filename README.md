# Tweet-Emotion-Recognition
This project been made on a recurrent neural network(RNN) and train it on a tweet emotion dataset to learn to recognize emotions in tweets. The dataset has thousands of tweets each classified in one of 6 emotions. This is a multi class classification problem in the natural language processing domain. USe of TensorFlow as our machine learning framework is done. A prior programming experience in Python is a need.  Optimization algorithms like gradient descent is also used but if you want to understand how to use the Tensorflow to start performing natural language processing tasks like text classification, this porject works on that level. Basic familiarity with TensorFlow is also needed.

# Learning Objectives
1. Using a Tokenizer in TensorFlow
2. Padding and Truncating Sequences
3. Creating and Training a Recurrent Neural Network
4. Using NLP and Deep Learning to perform Text Classification

# TensorFlow
TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google.

# Emotion Dataset
This is a dataset that can be used for emotion classification. It has already been preprocessed based on the approach described in our paper. It is also stored as a pandas dataframe and ready to be used in an NLP pipeline.

Note that the version of the data provided here corresponds to a six emotions variant that's meant to be used for educational and research purposes.

# Download
Hugging Face: https://huggingface.co/datasets/emotion

Download link: https://www.dropbox.com/s/607ptdakxuh5i4s/merged_training.pkl

Papers with Code Public Leaderboad: https://paperswithcode.com/sota/text-classification-on-emotion

# Notebooks
Here is a notebook showing how to use it for fine-tuning a pretrained language model for the task of emotion classification.

Here is another notebook which shows how to fine-tune T5 model for emotion classification along with other tasks.

Here is also a hosted fine-tuned model on HuggingFace which you can directly use for inference in your NLP pipeline.

Feel free to reach out to me on Twitter for more questions about the dataset.

# Usage
The dataset should be used for educational and research purposes only. If you use it, please cite:

@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}

# Research Opportunities
We are expanding this dataset to include more languages. If you would like to know more about this research project, feel free to reach out to me on Twitter.
