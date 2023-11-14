# Mini-Project
# ANALYSIS OF THE DETAILS OF A PERSON
# Aim:
 Analysis of the details of a person.
 
# ALGORITHM:
Step:1  Importing necessary packages.

Step:2  Read the data set.

Step:3  Execute the methods.

Step:4  Run the program.

Step:5  Get the output.

# CODE AND OUTPUT:
```
NAME: Ragavendran A
REG NO: 212222230114
```
```python
import pandas as pd
df = pd.read_csv("addresses.csv")

df.head(4)
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/f0f55571-bad6-4ba8-ab6c-c7c76ed18fe4)

```python
df.info()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/470ecdcd-aa18-451e-9863-a7949b67a4f5)


```python
df.dropna(how='all').shape
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/38665a05-a4f9-4619-985f-013c8d280419)

```python
df.fillna(0)
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/929eec23-1a02-4da0-ab7d-aa1e8f021426)


```python
df.fillna(method='bfill')
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/4ae410e4-8a37-4613-8d5d-a63e8fa028d8)


```python
df.duplicated()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/c499075c-b2ca-4ac7-bb81-21e1d7bb13f7)


```python
exp = [13,23,28,12,5,9,31,26,10,19,22,24,29,4,25,30]
af=pd.DataFrame(exp)
af
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/e274c952-a0db-40e6-bb26-731bb1605354)


```pyhton
q1=af.quantile(0.25)
q2=af.quantile(0.5)
q3=af.quantile(0.75)
iqr=q3-q1

low=q1-1.5*iqr
low
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/c9f2e941-a243-4459-baed-c098adf3b79e)

```python
high=q1+1.5*iqr
high
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/1a40f1b1-d60b-45a5-9af4-d44a1d486d19)


```python
sns.boxplot(data=af)
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/0bd91475-fd69-4c6a-abcd-4d6094ea8ca9)


```python
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("addresses.csv")


plt.figure(figsize=(8, 4))
data['Desig'].value_counts().plot(kind='bar')
plt.title('Distribution of Desig')
plt.xlabel('Desig')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/6d9f61fd-e35a-4786-b8ca-a94db8d42773)


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data, hue="Desig")
plt.show()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/d9c6c278-8969-4bb0-94c3-2ed1823ad720)


```python
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/d6657d5c-7cba-4816-99ef-97371f008c35)


```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

numerical_features = ['ID']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
data
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/e4d70e65-a059-4410-8053-8da823eac1f7)


```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
columns_to_scale = ['ID']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
data
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/0a54f657-4e63-4f9a-bbc4-5fc82b5767de)


```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
data[['ID']] = scaler.fit_transform(data[['ID']])
data
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/4ec68fbb-2f4e-4d59-877a-2d48f5105c25)


```python
data.skew()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/c1835689-6cff-4fea-b7d0-0eb062ad896f)


```python
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np

np.log(df["ID"])
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/648094f3-8ce4-4a44-9b25-85eed638ea1f)


```python
np.sqrt(df["ID"])
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/0d92dd99-2c11-4fd6-9128-54a3fa24a4c6)


```python
sm.qqplot(df['ID'],line='45')
plt.show()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/81a92361-4689-40ab-aba7-bb63ab267a4d)


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
plt.hist(data['ID'], bins=10, color='skyblue', edgecolor='black')
plt.title('Position')
plt.xlabel('ID')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/9257423d-d348-48c8-a391-dfd4dfe9eff6)


```python
plt.figure(figsize=(8, 4))
sns.boxplot(data=data, x='ID', color='lightcoral')
plt.title('Position Boxplot')
plt.xlabel('ID')
plt.show()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/059fb79a-629d-4760-a2fa-15c72a02bd47)


```python
plt.figure(figsize=(10, 4))
sns.countplot(data=data, x='Desig', palette='Set2')
plt.title('Desig Counts')
plt.xlabel('Desig')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/Leann4468/Mini-Project/assets/121165979/148525f2-f194-4dde-a262-a8e159dd3a6e)




# Result:
Hence the program to analyze the data set using data science is applied sucessfully.
