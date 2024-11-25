# Diet Type Classification Using Macronutrients

## Overview
This project focuses on building a machine learning model to predict a recipe's diet type (`Diet_type`) based solely on its macronutrient profile (`Protein(g)`, `Carbs(g)`, and `Fat(g)`). By analyzing nutritional data, the model provides insights into how these macronutrients influence diet classification, supporting applications such as recipe categorization or personalized meal planning.

The decision tree algorithm is employed to train and evaluate the model, offering interpretability through a tree-based structure. Additionally, the project includes visualizations of the decision tree, highlighting the decision-making process.

---

## Aim of the Model

### **Objective**
To classify recipes into diet types (`DASH`, `Keto`, `Mediterranean`, `Paleo`, `Vegan`) using only their macronutrient content.

### **Key Goals**
- Provide a simple yet interpretable classification of diet types.
- Analyze the importance of macronutrients (`Protein`, `Carbs`, `Fat`) in diet classification.
- Visualize the decision-making process of the trained model.

---
## Data
The data used to make this model was taken from github 
https://github.com

## Technologies Used

### **Programming Language**: 
- Python

### **Libraries**:
- **pandas**: For data manipulation and preprocessing.
- **sklearn**:
  - `DecisionTreeClassifier`: For training the classification model.
  - `train_test_split`: For splitting the dataset into training and testing subsets.
  - `plot_tree`: For visualizing the decision tree.
  - `accuracy_score`: For evaluating the model's performance.
- **matplotlib**: For generating visualizations.

---
## Algorithm

### **Decision Tree Classifier**

#### **Why Decision Trees?**
- Decision Trees are simple and interpretable models that split the dataset based on feature thresholds, creating a tree-like structure.
- They allow users to see how features like `Protein`, `Carbs`, and `Fat` contribute to classifying diet types.

#### **Steps**:
1. The model learns patterns in macronutrient data to predict the diet type during training.
2. At each split, the tree selects the feature that provides the highest information gain (or reduces impurity the most).
3. Leaf nodes represent the final predicted diet type.

---

## Performance Metrics

### **Accuracy**:
- The model achieved an accuracy of **38.81%** on the test dataset.
- This accuracy reflects how often the model correctly predicts the diet type for unseen recipes.

### **Error Metric: Mean Squared Error (MSE)**:
- MSE measures the average squared difference between the predicted and actual values (encoded classes for diet types).
- The lower the MSE, the better the model’s performance.
##  Visualize the Decision Tree

![image](https://github.com/user-attachments/assets/854376fd-442f-4a77-8d65-7fd149b1bc25)

## Code
```python
# Calculate Mean Squared Error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

## **Code**:
# Adjust features to include only macronutrients
X = data_cleaned[['Protein(g)', 'Carbs(g)', 'Fat(g)']]
y = data_cleaned['Diet_type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42, max_depth=3)
dt_model.fit(X_train, y_train)

# Evaluate the model
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Decision Tree Accuracy: {accuracy:.2f}")

# Map the encoded Diet_type values back to their original labels
diet_type_labels = label_encoders['Diet_type'].inverse_transform(sorted(set(y)))

# Visualize the Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=diet_type_labels,  # Use diet type labels instead of numeric classes
    filled=True,
    rounded=True
)
plt.title("Decision Tree Rooted on Macronutrients for Predicting Diet_type")
plt.show()



