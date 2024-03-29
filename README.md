# NLPCC2023-Task9-Track1-1st
Code for the NLPCC2023 paper [Adversarial Training and Model Ensemble for User Feedback Prediciton in Conversation System](https://link.springer.com/chapter/10.1007/978-3-031-44699-3_33).  
Our system ranks **first** in NLPCC 2023 shared Task 9 Track 1 [User Feedback Prediciton](https://github.com/XiaoMi/nlpcc-2023-shared-task-9).
## Abstract
Developing automatic evaluation methods that are highly correlated with human assessment is crucial in the advancement of dialogue systems. User feedback in conversation system provides a signal that represents user preferences and response quality. The user feedback prediction (UFP) task aims to predict the probabilities of likes with machine-generated responses given a user query, offering a unique perspective to facilitate dialogue evaluation. In this paper, we propose a powerful UFP system, which leverages Chinese pre-trained language models (PLMs) to understand the user queries and system replies. To improve the robustness and generalization ability of our model, we also introduce adversarial training for PLMs and design a local and global model ensemble strategy. **Our system ranks first in NLPCC 2023 shared Task 9 Track 1 (User Feedback Prediction)**. The experimental results show the effectiveness of the method applied in our system.
## Strategy Overview
We used the following strategies to address this task：
* Fine-tune five Chinese Pretrained Language Models: **RoBERTa-wwm-large,ERNIE-3.0-xbase,ERNIE-3.0-base,RoBERTa-MRC-wwm-large and MacBERT-large**.
* PGD Adversarial Training.
* Multi-Sample Dropout.
* Data augmentation.
* Linear Regression for Blending Ensemble.
## Experiment Results
![strategy-results](https://github.com/Junlong-Wang/NLPCC2023-Task9-Track1-1st/blob/main/%E7%AD%96%E7%95%A5%E6%8C%87%E6%A0%87.png)
![ensemble-results](https://github.com/Junlong-Wang/NLPCC2023-Task9-Track1-1st/blob/main/%E9%9B%86%E6%88%90%E6%8C%87%E6%A0%87.png)
