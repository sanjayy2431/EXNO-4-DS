# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING :
```
from google.colab import files
files.upload()
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, RobustScaler

# Load BMI dataset
df = pd.read_csv("bmi.csv")

# Display dataset
print("BMI Dataset:")
display(df.head())

# Remove missing values
df = df.dropna()

# Select Height and Weight
X = df[['Height', 'Weight']]

print("Maximum Height:", X['Height'].max())
print("Maximum Weight:", X['Weight'].max())

# ---------------- MIN-MAX SCALING ----------------
minmax = MinMaxScaler()
minmax_scaled = minmax.fit_transform(X)

print("\nMin-Max Scaled Data:")
display(pd.DataFrame(minmax_scaled, columns=['Height', 'Weight']).head())

# ---------------- STANDARD SCALING ----------------
standard = StandardScaler()
standard_scaled = standard.fit_transform(X)

print("\nStandard Scaled Data:")
display(pd.DataFrame(standard_scaled, columns=['Height', 'Weight']).head())

# ---------------- NORMALIZATION ----------------
normalizer = Normalizer()
normalized = normalizer.fit_transform(X)

print("\nNormalized Data:")
display(pd.DataFrame(normalized, columns=['Height', 'Weight']).head())

# ---------------- MAX ABS SCALING ----------------
maxabs = MaxAbsScaler()
maxabs_scaled = maxabs.fit_transform(X)

print("\nMaxAbs Scaled Data:")
display(pd.DataFrame(maxabs_scaled, columns=['Height', 'Weight']).head())

# ---------------- ROBUST SCALING ----------------
robust = RobustScaler()
robust_scaled = robust.fit_transform(X)

print("\nRobust Scaled Data:")
display(pd.DataFrame(robust_scaled, columns=['Height', 'Weight']).head())
```
# OUPUT :
<img width="322" height="397" alt="image" src="https://github.com/user-attachments/assets/485baa49-5993-44bb-a5fc-ae7d63d3d3f7" />


<img width="181" height="381" alt="image" src="https://github.com/user-attachments/assets/081de1eb-a5fe-468c-9282-8f8ff7fb4b6a" />



<img width="165" height="367" alt="image" src="https://github.com/user-attachments/assets/aaa25ed9-1d2e-40bb-bca3-5c8e72776f6f" />

# RESULT:
       # INCLUDE YOUR RESULT HERE
