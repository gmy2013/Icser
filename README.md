<!-- # Icser
Interpretation-based Code Summarization

Our paper is available on -->


# Interpretation-based Code Summarization (ICPC 2023)

Our paper is available on [ResearchGate](https://shangwenwang.github.io/files/ICPC-23A.pdf).


## Introduction

Code comment, i.e., the natural language text to
describe the semantic of a code snippet, is an important way
for developers to comprehend the code. Recently, a number
of approaches have been proposed to automatically generate
the comment given a code snippet, aiming at facilitating the
comprehension activities of developers. Despite that state-of-the-art
 approaches have already utilized advanced machine learning
techniques such as the Transformer model, they often ignore
critical information of the source code, leading to the inaccuracy
of the generated summarization. In this paper, to boost the
effectiveness of code summarization, we propose a two-stage
paradigm, where in the first stage, we train an off-the-shelf
model and then identify its focuses when generating the initial
summarization, through a model interpretation approach, and
in the second stage, we reinforce the model to generate more
qualified summarization based on the source code and its focuses.
Our intuition is that in such a manner the model could learn to
identify what critical information in the code has been captured
and what has been missed in its initial summarization, and
thus revise its initial summarization accordingly, just like how a
human student learns to write high-quality summarization for a
natural language text. Extensive experiments on two large-scale
datasets show that our approach can boost the effectiveness of
five state-of-the-art code summarization approaches significantly.


## Get Started

Install PyTorch. 
The code has been tested with CUDA 11.2/CuDNN 8.1.0, PyTorch 1.8.1.

Prepare the dataset through [CodeSearchNet](https://github.com/github/CodeSearchNet). 



## Citation
```
@article{genginterpretation,
  title={Interpretation-based Code Summarization},
  author={Geng, Mingyang and Wang, Shangwen and Dong, Dezun and Wang, Haotian and Cao, Shaomeng and Zhang, Kechi and Jin, Zhi}
}
```




