

## File structure:
```
├── model.py: ResNet model construction 
├── train.py: Training script 
├── predict.py: Single image prediction script
```

●The sentiment classification system is a technology that analyzes images of faces to determine the emotional state of the person in the image, such as happy, sad, angry, and more.

●It takes an input image containing facial expressions.

●Attributes the image to its corresponding emotional label.

Project Information
We are building ResNet-34 models to predict facial expressions.
Assessing the performance and comparing different models.
Classification performance metrics include: accuracy, precision, recall, and f-1 score.

Dataset : FER-2013
Used by Kaggle in one of the competitions.
Contains 48 x 48 grayscale labeled images of facial expressions.
Includes approximately 29K examples as training set and 7K sample images for test set.
Hugely imbalanced.

The number of samples corresponding to each expression in the training sample data on the left.
The picture on the right predicts the number of samples corresponding to each expression in the sample data.
![alt text](http://url/to/img.png)
