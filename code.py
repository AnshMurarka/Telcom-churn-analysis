import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
# To preprocess the data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# machine learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
#for classification tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
#Cross validation
from sklearn.model_selection import StratifiedKFold
# pipeline
from sklearn.pipeline import Pipeline
# metrics
from sklearn.metrics import accuracy_score,mean_squared_error
# ignore warnings   
import warnings
warnings.filterwarnings('ignore')
# to visualize data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix
#To save the model
import joblib

df = pd.read_csv() # Insert link
pd.set_option('display.max_columns', None)
df.head()

# Converting TotalCharges to numeric datatype
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.isnull().sum()
df = df.drop(['customerID'], axis = 1)
df.head()
#Finding the missing 11 data
df[np.isnan(df['TotalCharges'])]
df.dropna(inplace=True)

#Check for unique values in categorical data
for col in df.columns:
    if df[col].dtype != 'int64' and df[col].dtype != 'float64':
        print(f'{col} : {df[col].unique()}')


#Data Visulaization
df["Churn"].value_counts().plot(kind='barh' , color = 'red');
print()
import matplotlib.ticker as mtick
# Define colors
colors = ['#4D3425', '#E4512B']
# Calculate churn percentage
churn_percent = (df['Churn'].value_counts() / len(df)) * 100
# Plot bar chart
fig, ax = plt.subplots(figsize=(8, 6))
churn_percent.plot(kind='bar', color=colors, rot=0, ax=ax)
# Format y-axis as percentage
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# Labels and title
ax.set_ylabel('% Customers', fontsize=14)
ax.set_xlabel('Churn', fontsize=14)
ax.set_title('Churn Rate', fontsize=14)
# Add percentage labels on bars
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 3, p.get_height() - 4,
            f"{p.get_height():.1f}%", fontsize=12, color='white', fontweight='bold')
plt.show()

# make plot for tenure
churned = df[df['Churn'] == 'Yes']
not_churned = df[df['Churn'] == 'No']

# Plotting
plt.figure(figsize=(10, 6))
plt.hist([churned['tenure'], not_churned['tenure']], bins=10, color=['red', 'blue'], label=['Yes', 'No'])
plt.title(' Tenure by Churn')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add text on top of bars
for rect in plt.gca().patches:
    height = rect.get_height()
    plt.gca().text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')

# make plot for MonthlyCharges
churned = df[df['Churn'] == 'Yes']
not_churned = df[df['Churn'] == 'No']

# Plotting
plt.figure(figsize=(10, 6))
plt.hist([churned['MonthlyCharges'], not_churned['MonthlyCharges']], bins=10, color=['red', 'blue'], label=['Yes', 'No'])
plt.title('MonthlyCharges by Churn')
plt.xlabel('MonthlyCharges')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add text on top of bars
for rect in plt.gca().patches:
    height = rect.get_height()
    plt.gca().text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')

# make plot for tenure
churned = df[df['Churn'] == 'Yes']
not_churned = df[df['Churn'] == 'No']

# Plotting
plt.figure(figsize=(10, 6))
plt.hist([churned['TotalCharges'], not_churned['TotalCharges']], bins=10, color=['red', 'blue'], label=['Yes', 'No'])
plt.title(' TotalCharges by Churn')
plt.xlabel('TotalCharges')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
# Add text on top of bars
for rect in plt.gca().patches:
    height = rect.get_height()
    plt.gca().text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')

#Calculating some essential data 
# Filter churned customers
churned_customers = df[df['Churn'] == 'Yes']
# Count customers using Fiber Optic
fiber_churned = churned_customers[churned_customers['InternetService'] == 'Fiber optic']
# Number of people who left due to poor fiber service
num_fiber_service = fiber_churned.shape[0]
# Total number of churned customers
total_churned = churned_customers.shape[0]
# Percentage of total churned customers who left due to poor fiber service
percentage = (num_fiber_service / total_churned) * 100
print(f"Total churned customers: {total_churned}")
print(f"Churned due to poor fiber service: {num_fiber_service}")
print(f"Percentage of churned customers due to poor fiber service: {percentage:.2f}%")

# Filter churned customers
churned_customers = df[df['Churn'] == 'Yes']
# Count customers using their plan tenure
month_churned = churned_customers[churned_customers['Contract'] == 'Month-to-month']
one_year_churned = churned_customers[churned_customers['Contract'] == 'One year']
two_year_churned = churned_customers[churned_customers['Contract'] == 'Two year']
# Number of people who left their respective plans
month_churned_people=month_churned.shape[0]
one_year_churned_people=one_year_churned.shape[0]
two_year_churned_people= two_year_churned.shape[0]
# Total number of churned customers
total_churned = churned_customers.shape[0]
# Percentage of total churned customers who left
percentage_month = ( month_churned_people / total_churned) * 100
percentage_one_year = ( one_year_churned_people / total_churned) * 100
percentage_two_year = ( two_year_churned_people / total_churned) * 100
print(f"Total churned customers: {total_churned}")
print(f"Churned from monthly contract: {month_churned_people}")
print(f"Percentage of churned customers from monthly contract: {percentage_month:.2f}%")
print(f"Churned from one year contract: {one_year_churned_people}")
print(f"Percentage of churned customers from one year contract: {percentage_one_year:.2f}%")
print(f"Churned from two year contract: {two_year_churned_people}")
print(f"Percentage of churned customers from two year contract: {percentage_two_year:.2f}%")

sns.set_context("paper",font_scale=1.1)
ax = sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'No') ],color="Red", shade = True);
ax = sns.kdeplot(df.MonthlyCharges[(df["Churn"] == 'Yes') ],ax =ax, color="Blue", shade= True);
ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Monthly Charges');
ax.set_title('Distribution of monthly charges by churn');

ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'No') ],color="Gold", shade = True);
ax = sns.kdeplot(df.TotalCharges[(df["Churn"] == 'Yes') ],ax =ax, color="Green", shade= True);
ax.legend(["Not Chu0rn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Total Charges');
ax.set_title('Distribution of total charges by churn');

#Using label encoder
def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series
df = df.apply(lambda x: object_to_int(x))
df.head()

# split data into X and y
X = df.drop('Churn', axis=1)
y = df['Churn']
# data into train and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42,stratify=y)
print(X_train.shape)
print(X_test.shape)

# Initialize an empty list to store model scores
model_scores = []
# Create a list of models to evaluate
models = [
    ('Random Forest', RandomForestClassifier(random_state=42),
        {'model__n_estimators': [50, 100, 200],
         'model__max_depth': [None, 10, 20]}),  # Add hyperparameters for Random Forest
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42),
        {'model__n_estimators': [50, 100, 200],
         'model__learning_rate': [0.05, 0.1, 0.5]}),  # Add hyperparameters for Gradient Boosting
    ('Support Vector Machine', SVC(random_state=42, class_weight='balanced'),
        {'model__C': [0.1, 1, 10],
         'model__gamma': ['scale', 'auto']}),  # Add hyperparameters for SVM
    ('Logistic Regression', LogisticRegression(random_state=42, class_weight='balanced'),
        {'model__C': [0.1, 1, 10],
         'model__penalty': ['l1', 'l2']}),  # Add hyperparameters for Logistic Regression
    ('K-Nearest Neighbors', KNeighborsClassifier(),
        {'model__n_neighbors': [3, 5, 7],
         'model__weights': ['uniform', 'distance']}),  # Add hyperparameters for KNN
    ('Decision Tree', DecisionTreeClassifier(random_state=42),
        {'model__max_depth': [None, 10, 20],
         'model__min_samples_split': [2, 5, 10]}),  # Add hyperparameters for Decision Tree
    ('Ada Boost', AdaBoostClassifier(random_state=42),
        {'model__n_estimators': [50, 100, 200],
         'model__learning_rate': [0.05, 0.1, 0.5]}),  # Add hyperparameters for Ada Boost
    ('XG Boost', XGBClassifier(random_state=42),
        {'model__n_estimators': [50, 100, 200],
         'model__learning_rate': [0.05, 0.1, 0.5]}),  # Add hyperparameters for XG Boost
    ('Naive Bayes', GaussianNB(), {})  # No hyperparameters for Naive Bayes]
best_model = None
best_accuracy = 0.0
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Iterate over the models and evaluate their performance
for name, model, param_grid in models:
    # Create a pipeline for each model
    pipeline = Pipeline([
        #('scaler', MinMaxScaler()),  # Feature Scaling
        ('model', model)])
    # Hyperparameter tuning using GridSearchCV
    if param_grid:
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv)
        grid_search.fit(X_train, y_train)
        pipeline = grid_search.best_estimator_
    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)
# Calculate accuracy score
    test_accuracy = accuracy_score(y_test, y_pred)
    rmse= np.sqrt(mean_squared_error(y_test, y_pred))
    # Append model name and accuracy to the list
    model_scores.append({'Model': name, 'Train Accuracy': train_accuracy, 'Test Accuracy': test_accuracy,'RMSE': rmse})
    # Convert the list to a DataFrame
    scores_df = pd.DataFrame(model_scores)
    # Print the performance metrics
    print("Model:", name)
    print("Train Accuracy:",train_accuracy.round(3))
    print("Test Accuracy:", test_accuracy.round(3))
    print("Test RMSE:",rmse.round(3))
    print()
    # Check if the current model has the best accuracy
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        bestmodel_trainaccuracy= train_accuracy
        best_model = pipeline
        best_model_rmse=rmse
        # Save the best model when it performs better than previous ones
        joblib.dump(best_model, 'best_model_pipeline.pkl')  # Save the best model
# Retrieve the overall best model
print("Best Model:")
print("Train Accuracy:", bestmodel_trainaccuracy)
print("Test Accuracy:", best_accuracy)
print("Test RMSE:",best_model_rmse)
print("Model Pipeline:", best_model, "with accuracy", best_accuracy.round(2))

# Define a color palette for the bars
colors = sns.color_palette('pastel', n_colors=len(scores_df))
# Create a bar plot of models and their scores
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Model', y='Test Accuracy', data=scores_df, palette=colors)
# Add text on each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), 
                textcoords='offset points')
plt.title('Model Scores')
plt.xlabel('Models')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# Load the saved best model pipeline
loaded_model = joblib.load('best_model_pipeline.pkl')
# Use the loaded model to make predictions
y_pred = loaded_model.predict(X_test)
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Plot confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Using ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
model = keras.Sequential([
    keras.layers.Dense(19, input_shape=(19,), activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])
# opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Initialize EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=10,         # Stop after 10 epochs if no improvement
    restore_best_weights=True,  # Restore the best model weights
    verbose=1)
model_history=model.fit(X_train, y_train,
                        epochs=100, validation_split=0.2, callbacks=[early_stopping])
model.save("model.h5")

plt.figure(figsize = (12, 6))

train_loss = model_history.history['loss']
val_loss = model_history.history['val_loss'] 
epoch = range(1,77)
sns.lineplot(x=epoch, y=train_loss, label = 'Training Loss')
sns.lineplot(x=epoch, y=val_loss, label = 'Validation Loss')
plt.title('Training and Validation Loss\n')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize = (12, 6))
train_loss = model_history.history['accuracy']
val_loss = model_history.history['val_accuracy'] 
epoch = range(1,77)
sns.lineplot(x=epoch,y= train_loss, label = 'Training accuracy')
sns.lineplot(x=epoch, y=val_loss, label = 'Validation accuracy')
plt.title('Training and Validation Accuracy\n')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

from keras.models import load_model
model = load_model("model.h5")
acc = model.evaluate(X_test, y_test)[1]
print(f'Accuracy of model is {acc}')

yp = model.predict(X_test)
yp[:5]
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


y_pred[:10]
y_test[:10]

from sklearn.metrics import confusion_matrix , classification_report
print(classification_report(y_test,y_pred))
import seaborn as sn
import tensorflow as tf
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
