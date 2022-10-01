# import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# constants
IRIS_CSV = './data/raw/iris.csv'

# data prep
df = pd.read_csv(IRIS_CSV)
X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=33)

pd.DataFrame(X_train).to_csv('./data/processed/x_train.csv', header=None, index=False)
pd.DataFrame(X_test).to_csv('./data/processed/x_test.csv', header=None, index=False)
pd.DataFrame(y_train).to_csv('./data/processed/y_train.csv', header=None, index=False)
pd.DataFrame(y_test).to_csv('./data/processed/y_test.csv', header=None, index=False)
