import numpy as np

class KNeighborsClassifier:

    def __init__(self, k):
        '''
        Takes k as number of neighbors
        '''
        self.k = k

    def fit(self, X_train, y_train):
        '''
        Takes X_train, y_train, adds to KNeighborsClassifier class
        '''
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        '''
        Takes X_test, returns y_pred predicted labels for X_test 
        '''
        def l2_norm(a, b):
            return np.sqrt(np.sum((a - b)**2))
        
        def find_neighbors(X_test):
            neighbors = []
            for i in range(self.X_train.shape[0]):
                dist = l2_norm(X_test, self.X_train[i])
                neighbors.append([i, dist, self.y_train[i]])
            neighbors = sorted(neighbors, key = lambda x: x[1])
            neighbors = neighbors[:self.k]
            return neighbors
        
        def voter(neighbors):
            freq = {}
            for i in neighbors:
                if i[2] not in freq:
                    freq[i[2]] = 1
                else:
                    freq[i[2]] += 1
            return sorted(freq.items(), key = lambda x:x[1], reverse = True)[0][0]
        
        y_pred = []
        for i in range(X_test.shape[0]):
            neighbors = find_neighbors(X_test[i])
            y_pred.append(voter(neighbors))
        y_pred = np.array(y_pred, dtype = int)
        return y_pred

    def score(self, y_pred, y_test):
        '''
        Takes y_pred predicted labels for X_test, y_test actual labels of X_test
        Returns accuracy of the model 
        '''
        return (y_pred == y_test).mean()
