import streamlit as st
import pandas as pd

import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score

warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')

_lock = RendererAgg.lock


st.set_page_config(layout="wide")

st.title('''CUSTOMER ANALYTICS''')

st.markdown("""
##

Predicting if purchase probability of a customer depends on age and Estimated salary where I use the following feature variables
- Age – Estimated Age of our customer 
- Estimated salary of the customer 
- and for my binary target variable Purchased where 1 indicates a purchase and 0 means a non-purchase 

Other Variables:
- User Id – This is the unique identifier of every customer that visits our store
- Gender – Sex of the customer

## Hypothesis
**Age**: The higher the age the higher chance making a purchase

**Estimated Salary**: Customers with a high income have a higher chance of making a purchase

""")

plt.style.use('fivethirtyeight')

df = pd.read_csv('social_network_ads.csv')

st.write('''## Exploratory Data Analysis''')
st.write('')
st.write('''#### General information''')
st.write('')

row1_1, row1_2, row1_3, = st.beta_columns((2, 1, 1))
with row1_1, _lock:
    st.write(df.sample(7))

with row1_2, _lock:
    st.write(df[['Age','EstimatedSalary']].describe())

with row1_3, _lock:
    st.write(df.groupby('Gender')[['EstimatedSalary']].describe())
    st.write(df.groupby('Gender')[['Age']].describe())

row2_1, row2_space_1, row2_2 = st.beta_columns((1,.1, 1))
with row2_1, _lock:
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    sns.boxplot(data=df,x='Gender',y='Age')
    ax.set_ylabel('Age')
    ax.set_xlabel('Gender')
    st.pyplot(fig)
    st.markdown("The female customers had higher variance in their age than men")

with row2_2, _lock:
    fig = plt.figure(figsize=(10,8))
    ax = fig.subplots()
    sns.boxplot(data=df,x='Gender',y='EstimatedSalary')
    ax.set_ylabel('Estimated Salary')
    ax.set_xlabel('Gender')
    st.pyplot(fig)
    st.markdown("The female customers had higher variance in their estimated salaries than men")


row3_1, row3_2 = st.beta_columns((1,1))
with row3_1, _lock:
    fig = plt.figure(figsize=(12,10))
    sns.scatterplot(data=df,x='Age',y='EstimatedSalary', hue='Gender')
    plt.title('Scatter plot for Age against Estimated Salary')
    st.pyplot(fig)
    st.markdown("It looks like the age of customers is not stronlgy correlated to the estimated salary. Which is justified since the pearson correlation coefficient is 0.1552 ")

with row3_2, _lock:
    fig = plt.figure(figsize=(12,10))
    sns.scatterplot(data=df,x='Age',y='EstimatedSalary', hue='Purchased')
    plt.title('Scatter plot for Age against Estimated Salary')
    st.pyplot(fig)
    st.markdown("It looks like the higher the salary the higher the chance a customer purchases a product and the older the customer is, the higher the chance he/she purchases a prodcut.")

## MODEL BUILDING
st.write('''## Model Building''')
X = df.iloc[:,[2,3]].values # Independent feature variables
y = df.iloc[:,4].values.reshape(-1,1) # sklearn expects a 2D array not a 1D array

std = StandardScaler()
X_std=std.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=0)

# fitting the train set
st.write('''
### Using various Machine Learning ALgorithms
 - Accuracy is the fraction of predictions our model got right.
 - Confusion Matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class.
    Following this order
    |True Positive|False Positive|
    |True Negative|False Negative|
 - Precision score How often the prediction is correct or when the model predicts a customer purchasing, customers will actually be purchasing  0% of the time
 - For customers who purchase in the test set, the model can identify a purchase (recall_score)%  of the time
 - A roc_auc_score clsoer to 1 is favourable, it tells how much the model is capable of distinguishing between classes.
''')
row4_1, row4_2, row4_3 = st.beta_columns((1, 1, 1))
with row4_1, _lock:
    st.write('#### Logistic Regression')
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    #y_proba = model.predict_proba(X_test)
    st.write(f"Accuracy score: {100*(accuracy_score(y_test, y_predict))}%")
    
    cnf_matrix = confusion_matrix(y_test,y_predict)
    class_names = [0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position('bottom')
    plt.tight_layout()
    plt.title('Confusion matrix',y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)
    st.write("Precision score: {:.2f}%".format(100*(precision_score(y_test,y_predict))))
    st.write("Recall score: {:.2f}%".format(100*(recall_score(y_test,y_predict))))
    st.write("Roc Auc Score: {:.2f}%".format(100*(roc_auc_score(y_test,y_predict))))
    #0.5 is worhtless

with row4_2, _lock:
    st.write('#### Softmax Regression')
    softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs",C=10)
    softmax_reg.fit(X_train,y_train)
    sft_pred = softmax_reg.predict(X_test)
    st.write(f"Accuracy score: {100*(accuracy_score(y_test,sft_pred))}%")

    cnf2_matrix = confusion_matrix(y_test,sft_pred)
    class_names = [0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.heatmap(pd.DataFrame(cnf2_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position('bottom')
    plt.tight_layout()
    plt.title('Confusion matrix',y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)

    st.write("Precision score: {:.2f}%".format(100*(precision_score(y_test,sft_pred))))
    st.write("Recall score: {:.2f}%".format(100*(recall_score(y_test,sft_pred))))
    st.write("Roc Auc Score: {:.2f}%".format(100*(roc_auc_score(y_test,sft_pred))))

with row4_3, _lock:
    st.write('#### Gaussian Naive Bayes')
    modelnb = GaussianNB()
    modelnb.fit(X_train, y_train)
    nb_predict = modelnb.predict(X_test)
    st.write(f"Accuracy score: {100*(accuracy_score(y_test, nb_predict))}%")
    
    cnf3_matrix = confusion_matrix(y_test,nb_predict)
    class_names = [0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    sns.heatmap(pd.DataFrame(cnf3_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position('bottom')
    plt.tight_layout()
    plt.title('Confusion matrix',y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    st.pyplot(fig)

    st.write("Precision score: {:.2f}%".format(100*(precision_score(y_test,nb_predict))))
    st.write("Recall score: {:.2f}%".format(100*(recall_score(y_test,nb_predict))))
    st.write("Roc Auc Score: {:.2f}%".format(100*(roc_auc_score(y_test,nb_predict))))


st.write('''## Target variable EDA''')

bought = df.loc[df['Purchased']==1]
no_buy = df.loc[df['Purchased']==0]
Gender = pd.crosstab(df['Gender'],df['Purchased'])

row5_1, row5_2, row5_3 = st.beta_columns((1, 1, 1))
with row5_1, _lock:
    st.write('Purchased')
    st.write(bought[['Age','EstimatedSalary']].describe())

with row5_2, _lock:
    st.write('Did not purchase')
    st.write(no_buy[['Age','EstimatedSalary']].describe())

with row5_3, _lock:
    st.write('0 represents non-purchasers whereas 1 purchases')
    st.write(Gender)

st.write('')

st.write('Purchased')
st.write(df.groupby('Purchased')[['Age']].describe())

st.write('Did not purchase')
st.write(df.groupby('Purchased')[['EstimatedSalary']].describe())

#Gender['Conversion_rate'] = [i/Gender.iloc[j,:].sum() for i in Gender.iloc[:,1] for j in Gender.columns]

Gender['Conversion_rate'] = [100*(i[1]/(i[0] + i[1])) for i in Gender.values]
Gender2 = Gender.drop(['Conversion_rate'], axis=1)

row6_1, row6_2= st.beta_columns((1, 1))
with row6_1, _lock:
    st.write(' 0 represents non-purchasers and 1 represents purchasers')
    Gender
with row6_2, _lock:
    st.write('Pearson correlation of the independent features')
    X = df.iloc[:,[2,3]]#Independent feature variables
    st.write(X.corr(method='pearson'))

st.write('')
row7_1,row7_space_2, row7_2 = st.beta_columns(
    (1,.1, 1))
with row7_1, _lock:
    st.write('### Correlation Matrix')
    matrix = df.loc[: , df.columns != 'User ID'].corr()
    fig = plt.figure()
    sns.heatmap(matrix, vmax=1, square=True, cmap="YlGnBu", annot = True)
    st.pyplot(fig)
with row7_2, _lock:
    st.write('')
    st.write('')
    st.write('')
    st.write('''
    The correlation between age and estimated salary is relatively low, insignificant interdependence in this values
    
    The correlation between estimated salary and the purchase incidence is positive but not really strong. One can 
    conclude that the higher the estimated salary the higher the chance a customer buys a product.

    The correlation between age and purchase incidence is strong and positive and enough to conclude that the older a customer is, 
    the more likely he or she will purchase product. This can be attributed to the fact that the older you are the higher the
    estimated salary. One assumption I can make is that the product in question is expensive and is preferred by older customers
    .
    ''')

st.write('''
    ### Conclusion
    
    There is enough evidence to accept the hypothesis that **The higher the age the higher chance  of making a purchase.**

    However there isn't enough evidence to accept the hypothesis that **Customers with a high income have a higher chance of making a purchase**
''')