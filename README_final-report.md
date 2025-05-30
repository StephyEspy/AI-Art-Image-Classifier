AI Art Image Classification

ECS 170 Fall 2024

Aaryan Mohta, Angela Lee, Gyeongseob Brian Lee, Jess Fong

Junhee Lee, Kaitlyn Vo, Lana Wong, Stephanie Espinoza

1. Introduction

AI continues to evolve in our modern society, sparking debates in creative circles about its impact on the integrity and authenticity of artistic expression. Distinguishing whether art is created solely by human capabilities or with AI assistance has become increasingly challenging, raising questions about ethical creativity within the artist community. To address this, our project aims to develop a model capable of differentiating AI-generated images from human-created artwork. Leveraging supervised machine learning with convolutional neural networks (CNNs), we train the model using a labeled dataset containing both types of art. This labeled dataset provides the necessary supervision for the model to identify and refine its classification capabilities through training.

2. Background

A couple of concepts to be familiar with are: (1) Supervised learning, (2) CNN, and (3) Image Classification. We will be using these concepts throughout our final project. Supervised learning is a type of machine learning where an algorithm learns to map inputs to desired outputs based on a dataset of labeled examples. The goal is for the algorithm to generalize this mapping so it can make accurate predictions on new, unseen data. A Convolutional Neural Network (CNN) is a type of deep learning model designed to process and analyze data with a grid-like structure, such as images. It uses convolutional layers to automatically learn spatial hierarchies of features (like edges, textures, and patterns) by applying filters to the input data. CNNs are widely used in tasks like image classification, object detection, and facial recognition.

Image classification is a computer vision task where the goal is to assign a label or category to an input image. 

3. Methodology

The methodology for our project involved three main components: data cleaning with exploratory data analysis, the application of convolution neural networks (CNNs), and the overall training and evaluating process. 

3.1 Data Cleaning and Exploratory Data AnalysisS

The dataset initially consisted of 105,000 AI-generated images and 50,000 human-made art images, sourced from Kaggle datasets, ‘AI-ArtBench’. The dataset was pre-divided into training and testing subsets, each containing separate directories for AI-generated and human-made images. There are a total of 30,000 images in the test set, 20,000 AI-generated images and 10,000 human-made art images respectively. To prepare for the data analysis, each image was encoded with a class label: 0 for AI-generated art and 1 for human-made art. This encoding facilitated seamless integration into the classification pipeline to ensure that the model could interpret the data effectively for binary classification.

(Figure 1.1 Human-made Arts)

(Figure 1.2 AI Generated Arts)

A critical challenge in dataset preparation was the significant class imbalance between AI-generated and human-made images, with 105,000 AI-generated images compared to 50,000 human-made images. While one straightforward approach to addressing this imbalance could have been to randomly remove 55,000 AI-generated images to match the number of human-made samples, this method carries notable limitations. Removing a substantial portion of the AI-generated data risks reducing the diversity of patterns and features in the dataset. Since AI-generated images can exhibit a wide range of styles, textures, and designs, discarding these images might lead to the loss of valuable information that could hinder the model’s ability to generalize effectively.

(Figure 2 Data Augmented Images)

Instead, data augmentation techniques were applied to the human-made images to match the count of AI-generated images. Augmentation techniques such as rotations, horizontal flips, and zooming were applied to create variations of the existing of human-made images.This approach preserved the diversity while simultaneously enhancing the variability of the human-made samples. The augmented dataset not only balanced the class distributions but also improved the model’s robustness by exposing it to a wide range of human-made art features during training. Additionally, all images were resized to 224 x 224 pixels and normalized to a range of [0,1] to ensure compatibility with the input requirements of the chosen models. 

3.2 CNN Model Selection and Assessment

For this classification task, the ResNet-50 pre-trained CNN model was employed to fasten the process and reduce computational demand. ResNet-50 was chosen as the primary model due to its well-established performance in image classification tasks, with its residual connections that mitigate the vanishing gradient problem during training. It can distinguish subtle differences in texture and pattern between AI-generated and human images. To validate the choice of ResNet-50, comparisons were conducted with EfficientNet. Each model was evaluated under identical conditions using the exact same augmented dataset, with accuracy and loss amount used to assess their performance. 

(Figure 3 ResNet Model Architecture)

Training utilized binary cross-entropy as the loss function and the Adam optimizer with an initial learning rate of 0.001. Regularization techniques, including batch normalization and dropout were implemented to enhance generalization and reduce overfitting. Additionally, early stopping was applied to halt training when validation loss ceased to improve, conserving computational resources.

Fine-tuning was performed to enhance model specificity for this task. This involved unfreezing the last 10 layers of the pre-trained ResNet-50 and retraining them with a lower learning rate of  This selective retraining allowed the model to adapt to the nuanced differences between AI-generated and human-made art while retaining the general features learned from the original dataset. The implementation leveraged TensorFlow and Keras for model development and relied on Python libraries such as NumPy, Pandas, and Matplotlib for data processing and visualization. Training was conducted on hardware equipped with a GPU, enabling efficient experimentation and fine-tuning.

Through this comprehensive methodology, challenges such as dataset imbalance and computational constraints were effectively addressed, resulting in a reliable and efficient classifier capable of distinguishing between AI-generated and human-made art.

4. Results 

To have a visual display of how well the classifier works, the training and validation accuracies were plotted and a confusion matrix was developed to view the training results.  

(Figure 4 ResNet-50 Model Train and Validation Accuracy Plots)

(Figure 5 EfficientNet Model Train and Accuracy Plots)

The dataset was trained with both the ResNet-50 and EfficientNet models. As shown in Figure 4, the training accuracy of the ResNet-50 model reached over 90% while the validation accuracy reached a similar percentage of about 88%. In Figure 5, it can be observed that the accuracy of the EfficientNet model greatly underperforms compared to the accuracy of ResNet-50, with the training accuracy reaching as high as about 67% and the validation accuracy reaching about 57%.  

Additionally, for the ResNet-50 model, the validation and training accuracies improved simultaneously in a positive direction across each epoch. This is evidence that the model is not overfitting with the data provided, further proving that the augmentations made to the human-created art data helps generalize the model to make more accurate predictions. On the other hand, the training accuracy of the EfficientNet model is erratic and disconnected from the validation accuracy. This could possibly be due to EfficientNet using a compound scaling strategy (balancing depth, width, and resolution), which might not align well with the dataset provided and cause the model to not learn effectively for our purposes. 

From these findings, the ResNet-50 model with fine-tuning was selected as the basis for the project. 

(Figure 6 Confusion Matrix)

Aftering training our ResNet model, a confusion matrix was created to give another visual representation of the results. 17,608 of the AI-generated images and 8,734 of the human-created artworks were classified correctly by the model out of the 30,000 images total–20,000 AI-generated images and 10,000 human-made art images. This means about 87.8% of the images were correctly classified, matching the validation accuracy from Figure 4. The incorrect classifications could be due to overlapping features that look similar between the AI-created and human-made art, so it would be a matter of identifying what exactly these features are to better train our classifier for future use. 

Thus, with the accuracy of the Resnet-50 model reaching about 90% and the Confusion Matrix showing a clear sign of the model labeling data correctly, the project is confirmed to be successful in creating a tool that distinguishes between human-created and AI-generated artwork. 

5. Discussion 

Throughout this project, we encountered several challenges and learned valuable lessons in the process. One notable limitation was the computational expense of training a deep learning model from scratch. Despite initial attempts, it became evident that our available hardware, including a GPU that frequently disconnected and a CPU with limited processing power, was insufficient for handling the extensive computational demands of training a custom image classification model. This prompted the decision to leverage the ResNet-50 pre-trained CNN model, which not only reduced training time but also enabled us to focus on fine-tuning for the specific task of distinguishing between AI-generated and human-created artwork.

The ResNet-50 model achieved a classification accuracy of approximately 90%, demonstrating its ability to generalize well without overfitting. Data augmentation and fine-tuning contributed to this success, making the model effective in identifying subtle differences in the artwork.

These results highlight the potential of pre-trained models in resource-constrained scenarios. Moving forward, addressing hardware limitations and exploring additional architectures could further enhance performance, extending the model's applicability to more complex tasks. This project underscored the value of transfer learning in building efficient and reliable classifiers.

6. Conclusion

To sum up, this project demonstrated the ability of advanced AI techniques to effectively classify AI-generated and human-made art, achieving high accuracy through careful data balancing and model tuning. Beyond its technical success, the project highlights the practical value of AI in areas such as digital authenticity, copyright protection and creative support in more general purposes. It serves as a testament to how AI can address real-world challenges while fostering a deeper ethical understanding of its capabilities and implications.


7. Contribution

Kaitlyn Vo - Worked on loading the dataset and running code  

Gyeongseob Brian Lee - Worked on EfficientNetB0 implementation and training & validating

Aaryan Mohta - Worked on training the classifier and testing different parameters

Lana Wong - Worked on the introduction and reporting results of our code

Junhee Lee - Reported our methodologies and reported results

Angela Lee - Provided background in the report and discussed project findings

Jess Fong - Organized report content and results into slide deck

Stephanie Espinoza Gutierrez - Organized report content and results into slide deck