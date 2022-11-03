from sklearn.datasets import load_iris
from utils import normalize, train_test_split

data = load_iris()
X = normalize(data.data)
y = data.target
X_names = data.feature_names
y_names = data.target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

if __name__ == '__main__':
    print(type(X), type(y))
    print(X.shape, y.shape)
    print(X, '\n', y)
    print(X_names, '\n', y_names)
