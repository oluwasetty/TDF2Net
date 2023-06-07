# -*- coding: utf-8 -*-
# import the necessary packages

class Classifier:
    @staticmethod
    
    def RF(train_ft, test_ft, y_train):
    
        #RANDOM FOREST
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators = 50, random_state = 42)
        
        # Train the model on training data
        model.fit(train_ft, y_train) #For sklearn no one hot encoding
        
        #Now predict using the trained model. 
        pred = model.predict(test_ft)
        
        return pred
    
    def NB(train_ft, test_ft, y_train):
    
        #Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        
        # Train the model on training data
        model.fit(train_ft, y_train) #For sklearn no one hot encoding
        
        #Now predict using the trained model. 
        pred = model.predict(test_ft)
        
        return pred
    
    def LR(train_ft, test_ft, y_train):
    
        #Logistic Regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        
        # Train the model on training data
        model.fit(train_ft, y_train) #For sklearn no one hot encoding
        
        #Now predict using the trained model. 
        pred = model.predict(test_ft)
        
        return pred
    
    def KNN(train_ft, test_ft, y_train):
    
        #K Nearest Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=3)
        
        # Train the model on training data
        model.fit(train_ft, y_train) #For sklearn no one hot encoding
        
        #Now predict using the trained model. 
        pred = model.predict(test_ft)
        
        return pred
    
    def DT(train_ft, test_ft, y_train):
    
        #Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth =3, random_state = 42)
        
        # Train the model on training data
        model.fit(train_ft, y_train) #For sklearn no one hot encoding
        
        #Now predict using the trained model. 
        pred = model.predict(test_ft)
        
        return pred
    
    def SVM_RBF(train_ft, test_ft, y_train):
    
        #Support Vector Machine with Radial Basis Function
        from sklearn.svm import SVC
        model = SVC(kernel='rbf')
        
        # Train the model on training data
        model.fit(train_ft, y_train) #For sklearn no one hot encoding
        
        #Now predict using the trained model. 
        pred = model.predict(test_ft)
        
        return pred
    
    def SVM_PK(train_ft, test_ft, y_train):
    
        #Support Vector Machine with Polynomial Kernel
        from sklearn.svm import SVC
        model = SVC(kernel='poly')
        
        # Train the model on training data
        model.fit(train_ft, y_train) #For sklearn no one hot encoding
        
        #Now predict using the trained model. 
        pred = model.predict(test_ft)
        
        return pred
