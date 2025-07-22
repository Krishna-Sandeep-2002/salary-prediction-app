# %% [code]
# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
import pandas as pd    #Pandas is a Python library used for working with data sets.It has functions for analyzing, cleaning, exploring, and manipulating data.
import numpy as np     #Numpy Python library is used for including any type of mathematical operation in the code. It is the fundamental package for scientific calculation in Python. 
from sklearn.model_selection import train_test_split    #train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets: for training data and for testing data. With this function, you don't need to divide the dataset manually. By default, Sklearn train_test_split will make random partitions for the two subsets.
from sklearn.impute import SimpleImputer  #SimpleImputer it replace the missing values by the strategy means mean or mode mean for numerical columns and mode for categorical columns
from sklearn.preprocessing import StandardScaler #StandardScaler is used for scalling the data and make the data unitless also minimizing the values
from sklearn.preprocessing import OneHotEncoder  
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score #Sklearn metrics are import metrics in SciKit Learn API to evaluate your machine learning algorithms.
import matplotlib.pyplot as plt #matplotlib, which is a Python 2D plotting library, and with this library, we need to import a sub-library pyplot. This library is used to plot any type of charts in Python for the code
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
#from sklearn.neural_network import MLPRegressor

def DataUnderstanding(DataFrame):
  """
  Parameters:DataFrame
  This function returns the first 5 rows, shape and datatype of each column
  """
  print("***"*300) 
  print("\033[1m" + DataUnderstanding.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)
  print("###The top 5 rows in the DataSet###\n",DataFrame.head(5)) 
  print("**"*50)
  print("###The shape of the DataSet###\n",DataFrame.shape)  
  print("**"*50)
  print("###Data types of the columns###\n",DataFrame.dtypes)
def Del_columns(DataFrame, Colm=None):
    """
    Parametrs:DataFrame,Colm is contains the columns are to be deleted
    This function delete the column and prints the before and after columns
    """
    print("***"*300) 
    print("\033[1m" + Del_columns.__doc__ + "\033[0m")    # print the docstring
    print("***"*300)
    print('Before dropping: ', DataFrame.columns)
    DataFrame.drop(columns=Colm,axis=1, inplace=True)
    print('After dropping: ', DataFrame.columns)
def convdtypes(DataFrame,Colm=None):
  """
  Parameters:DataFrame,Colm is the columns which are to be in category
  this function will convert the columns data type to category
  """
  print("***"*300) 
  print("\033[1m" + convdtypes.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)
  print("Before Convertion of Datatypes:",DataFrame.dtypes)
  print("***"*50)
  print("***"*50)
  for i in Colm:
    DataFrame[i]=DataFrame[i].astype("category")
  print("After Convertion of Datatypes:",DataFrame.dtypes)
def Null_Unique_vals(DataFrame):
  """
  Parametrs:DataFRame,Colm is the columns to check for null values 
  this fucntion will checks the null values and unique values returns its count
  """
  print("***"*300) 
  print("\033[1m" + Null_Unique_vals.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)
  null_counts = DataFrame.isnull().sum()
  unique_counts = DataFrame.nunique()
  result_df = pd.DataFrame({'Column': null_counts.index, 'Null Values': null_counts.values, 'Unique Values': unique_counts.values})
  return result_df
def Unique_Vals(DataFrame, Colm=None):
  """
  Parameters: DataFrame , Colm is the columns of the dataframe 
  this function will return the unique values 
  
  """
  for i in  Colm:
    print('Number of unique values in',i,'is ', DataFrame[i].nunique(),'\n') #returns the number of unique values for each column.
    print(DataFrame[i].value_counts())
    print("**"*50)
    print("**"*50)
  
def X_Y_df(DataFrame,target_col):
  """
  This function will divide the dataset into X and y
  
  """
  print("***"*300) 
  print("\033[1m" + X_Y_df.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)
  y = DataFrame[target_col]
  X = DataFrame.drop(columns=target_col,axis=1)
  print("Shape of X:",X.shape)
  print("Shape of y:",y.shape)
  return X,y
def Train_Test_df(X,y):
  """
  This fucntion will  split the train and test test size 30%
  """
  print("***"*300) 
  print("\033[1m" + Train_Test_df.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)
  X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.3,random_state=123)
  print("The Shape of X_train is:",X_train.shape)
  print("The Shape of X_test is:",X_test.shape)
  print("The Shape of y_train is:",y_train.shape)
  print("The Shape of y_test is:",y_test.shape)
  return X_train,X_test,y_train,y_test
def num_cat_df(DataFrame):
    """
    This function will divide the data into num columns and cat columns
    """
    print("***"*300) 
    print("\033[1m" + num_cat_df.__doc__ + "\033[0m")    # print the docstring
    print("***"*300)
    #num_df=DataFrame.select_dtypes(include=["int64","float64"])
    #cat_df=DataFrame.select_dtypes(include=["category"])
    num_df = DataFrame.select_dtypes(include=['int64','float64'])          # here wee are selecting the columns which are in int and float datatyoe then wee store in DataFrame
    cat_df = DataFrame.select_dtypes(include=['category'])   
    print("The shape of Num_df is:",num_df.shape)
    print("The shape of Cat_df is:",cat_df.shape)  
    return num_df,cat_df                                
def get_dummies(df, cols):  
  """
  In this function, we perform one-hot encoding on specified categorical columns.
  It handles unknown categories safely and drops the first dummy to avoid dummy variable trap.
  """
  print("***" * 100) 
  print("\033[1m" + get_dummies.__doc__ + "\033[0m")
  print("***" * 100)

  # Initialize encoder
  enc = OneHotEncoder(handle_unknown='ignore', drop='first')

  # Fit only on selected columns
  enc = enc.fit(df[cols])

  # Transform only the same columns!
  enc_df = pd.DataFrame(enc.transform(df[cols]).toarray())

  # Give proper column names
  enc_df.columns = enc.get_feature_names_out(input_features=cols)

  print('After Performing Dummification on Categorical Columns:\n', enc_df.head(), '\n')
  return enc_df, enc

def Standard_Scaler(X_train_num,X_test_num): 
  """
  This is the Function Written in ml_functions.py File.
  In this Function I'm Standardizing the Numerical Data By Using 
  Standard_Scaler.
  """ 
  print("***"*300) 
  print("\033[1m" + Standard_Scaler.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                                                   # in this function we are doing Standardization for Scaling the data and convert columns into unit less on numerical columns
  scaler = StandardScaler()                                                             # StandardScaler() is the method using for Standardizing data i.e., calculating Z-score
  scaler=scaler.fit(X_train_num)                                                        # in fit() it calculates overall mean , SD values 
  X_train_num=pd.DataFrame(scaler.transform(X_train_num),columns=X_train_num.columns)   # transform(data) method is used to perform scaling using mean and std dev calculated using the . fit() method.
  X_test_num=pd.DataFrame(scaler.transform(X_test_num),columns=X_test_num.columns)
  print("After Standardizing X_train_num  Shape is:\n",X_train_num.shape,'\n')
  print("**"*50)
  print("After Standardizing X_test_num Shape is:\n",X_test_num.shape,'\n')
  return X_train_num,X_test_num,scaler

def concating_numcat_df(num_df,cat_df):
  """
  Combining the standardized numerical columns and categorical columns
  """
  print("***"*300) 
  print("\033[1m" + concating_numcat_df.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)
  df = pd.concat([num_df,cat_df],axis=1,join='inner')
  print(df.head())
  return df

# %% [code]
