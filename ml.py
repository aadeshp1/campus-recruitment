import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
placement = pd.read_csv("Placement_Data_Full_Class.csv")
print(placement.head(10))
placement_copy=placement.copy()
print(placement_copy.shape)
print(placement_copy.dtypes)
print(placement_copy.isnull().sum())
placement_copy['salary'].fillna(value=0 , inplace = True )
print(placement_copy.isnull().sum())
placement_copy.drop(['sl_no','ssc_b','hsc_b'], axis = 1 , inplace = True)
print(placement_copy.head())

plt.figure(figsize = (15,10))

ax = plt.subplot(221)
plt.boxplot(placement_copy['ssc_p'])
ax.set_title('Secondary School Percentage')


ax = plt.subplot(222)
plt.boxplot(placement_copy['hsc_p'])
ax.set_title('Higher secondary Percentage')

ax = plt.subplot(223)
plt.boxplot(placement_copy['degree_p'])
ax.set_title('UG Percentage')

ax = plt.subplot(224)
plt.boxplot(placement_copy['etest_p'])
ax.set_title('Employability Percentage')

