# ECS170 Project Proposal: Distinguishing AI-Generated Art


## Project Type
This is a system implementation project where we will be studying supervised machine learning through a classification CNN model. The learning signal guiding the algorithm comes from a dataset of paired input-output examples. Each artwork serves as the input, and the corresponding label– indicating whether it is AI-generated or human-made acts as the output. This labeled dataset provides the necessary supervision for the model to identify and learn the patterns and relationships with the data through training.

## Project Value
The value of our project lies in creating a tool for artists, collectors, and researchers to identify authentic works. According to the Academy of Animated Art, “74% of artists say that they believe AI artwork to be unethical.” Helping people identify AI-generated art would provide benefits to artists and people searching for real art. While similar solutions exist, this project updates the approach to reflect current AI art trends, contributing to the broader discussion on AI’s impact in creative fields.

## Computing Resources
We will use several Kaggle datasets to detect AI-generated artwork. The work involves image preprocessing and model training. We plan to use pre-trained models, but if we train from scratch, a high-performance GPU will be necessary. While we could handle initial tasks locally, cloud computing resources are preferred for faster processing and storage. These services offer free tiers but will require time to set up, which we are prepared to budget for as needed. We will also focus on optimizing models within these constraints and attempt to use multiple statistical methods for our model efficiency. 

## Scaffolding
Our scaffolding is a mixture of pre-classified Kaggle datasets, research papers, articles, and existing GitHub projects. A research paper titled [AI vs. AI: Can AI Detect AI-Generated Images?](https://www.mdpi.com/2313-433X/9/10/199) focuses on developing a model that detects AI-generated art using convolutional neural networks. We plan to use some of the techniques used in this research paper to replicate/improve the results of the model. We will also take inspiration from [“AI vs Real: Art Style Classification”](https://github.com/2spi/ai-v-real/), a project that develops an image classification model using CNNs. It has a high recall and precision rate, and we will aim to reproduce these metrics with our datasets. We will also present the probability of the image being AI-generated with two separate models on different datasets. 

![System Diagram](./figures/figure1-1)
**Figure 1.1:** Diagram of General Layout of Project

![System Diagram](./figures/figure1-2)
**Figure 1.2** CNN Layout

## Risk Assessment
One risk is potential biases that could be present in the datasets, i.e., overrepresentation of certain art styles that could skew the model’s performance for other types of artwork. To mitigate this, we could focus on one specific art style, i.e, Expressionism via assigning higher weights to other art styles. Another risk is the ethical impact our model could have on artists with false positives/negatives.” Our initial approach is to classify as “yes/no” if an artwork is AI-generated. If we see unreliable results, we can switch to providing a probability percentage of an artwork being AI-generated. 

## Project Scope
We will be implementing 1 model using a convolution layer and pooling layer within CNN, that classifies between real and AI-generated art. We will train the model on 2 datasets ([AI-ArtBench](https://www.kaggle.com/datasets/ravidussilva/real-ai-art), [Detecting AI-generated Artwork](https://www.kaggle.com/datasets/birdy654/detecting-ai-generated-artwork)) and compare the results from the model on the datasets. There are key data points that distinguish AI-generated art from human-made art: Texture Analysis, Color Distribution, Patterns and Contours, and Artwork metadata. Unlike human-made art, AI art may have distorted and unnatural textures, more algorithmic color placement, perfect symmetry, and a lack of completion date and artist information. 

## Presentation
Results will be in a slideshow depicting pairs of art images with the same subjects where one is real and one is AI-generated. To get these photo inputs to demonstrate that our classifier works, we will utilize an AI image generator using descriptors of a real image we already have to generate an AI counterpart. We plan to collect data on the features mentioned above for each image and present it in a chart where we can draw comparisons between the two images before demonstrating what the ML classifier outputs as the result for each image.
