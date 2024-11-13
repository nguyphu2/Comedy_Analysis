import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('comedy_data.csv')

#features: all columns starting from column 3
X = df.iloc[:,3:]

#target: ground truths from column 2

Y = df.iloc[:,2]

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

scaler = StandardScaler()

#standard scaling x train and x test
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaler = pd.DataFrame(
X_train_scaler, 
    columns=[
        'length', 'intensity_mean', 'intensity_std', 'intensity_min', 'intensity_max',
        'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 
        '1_mean', '1_std', '1_min', '1_max', '2_mean', '2_std', '2_min', '2_max', 
        '3_mean', '3_std', '3_min', '3_max', '4_mean', '4_std', '4_min', '4_max', 
        '5_mean', '5_std', '5_min', '5_max', '6_mean', '6_std', '6_min', '6_max', 
        '7_mean', '7_std', '7_min', '7_max', '8_mean', '8_std', '8_min', '8_max', 
        '9_mean', '9_std', '9_min', '9_max', '10_mean', '10_std', '10_min', '10_max', 
        '11_mean', '11_std', '11_min', '11_max', '12_mean', '12_std', '12_min', '12_max'
    ]
)

X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train_scaler.columns)


print('hi')
#machine model parameters
dt_classifier = DecisionTreeClassifier(random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
svm_classifier = SVC(kernel='rbf', random_state=42)

results={}
classifiers = {
    'Decision Tree': dt_classifier,
    'KNN': knn_classifier,
    'SVM': svm_classifier
}
#going through all the classifiers and training and getting predictions
for name, clf in classifiers.items():
    clf.fit(X_train_scaler,Y_train)
    
    
    y_pred = clf.predict(X_test_scaled)
    results[name] = {
        'predictions':y_pred,
        'report':classification_report(Y_test,y_pred),
        'confusion_matrix': confusion_matrix(Y_test,y_pred)

    }
    cv_scores = cross_val_score(clf, X_train_scaler, Y_train, cv=5)

    results[name]['cv_scores'] = cv_scores.mean()
    print(name, "Results:")
    print(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print("\nClassification Report:")
    print(classification_report(Y_test, y_pred))

def plot_confusion_matrices(results):
    container, plots = plt.subplots(1,3, figsize=(15,5))
    for index, (name,value) in enumerate(results.items()):
        ConfusionMatrixDisplay(value['confusion_matrix'],display_labels=[-1,0,1]).plot(ax=plots[index], cmap='Blues', colorbar=False)
        plots[index].set_title(f'{name} Confusion Matrx')
        plots[index].set_xlabel('Predicted')
        plots[index].set_ylabel('Actual')
    plt.show()

def plot_models_performance(results):
    model_names = []
    cv_means = []
    cv_stds = []
    
    # Loop through each model's results
    for name, value in results.items():
        model_names.append(name)  
        cv_means.append(value['cv_scores'].mean())  
        cv_stds.append(value['cv_scores'].std())  

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
    plt.title('Model Performance Comparison')
    plt.ylabel('Cross-validation Score')
    plt.ylim(0, 1)
    plt.show()


def finding_best_models(results):
    best_performance = 0
    best_model_name = ""  
    
    for model_name, value in results.items():
        current_performance = value['cv_scores']
        
        if current_performance > best_performance:
            best_performance = current_performance
            best_model_name = model_name
    
    print(f"Best Performing Model: {best_model_name} with an average Cross Validation Score of: {best_performance:.3f}")



plot_models_performance(results)
plot_confusion_matrices(results)
finding_best_models(results)