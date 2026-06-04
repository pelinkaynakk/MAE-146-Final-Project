import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# load data
X_train = np.loadtxt("train/X_train.txt")
y_train = np.loadtxt("train/y_train.txt")
X_test  = np.loadtxt("test/X_test.txt")
y_test  = np.loadtxt("test/y_test.txt")

# standardize
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# lambda values (sklearn uses C = 1/lambda)
lambdas = [0.01, 0.1, 1, 10, 100, 1000]

# logistic regression
logreg_val = []
logreg_train = []
for lam in lambdas:
    model = LogisticRegression(C=1/lam, max_iter=2000)
    logreg_val.append(cross_val_score(model, X_train, y_train, cv=5).mean())    # cv=5 splits the training data into 5, trains on 4 and validates on 1 left out, rotates so each part is held out once, and averages the 5 accuracies to score each lambda, so the best lambda is the one with highest avg
    model.fit(X_train, y_train) # trains data
    logreg_train.append(model.score(X_train, y_train)) # measures how successful

best_lambda_logreg = lambdas[np.argmax(logreg_val)]
print("Logistic regression best lambda:", best_lambda_logreg)

# linear SVM
svm_val = []
svm_train = []
for lam in lambdas:
    model = LinearSVC(C=1/lam, max_iter=5000) # higher iterations for SVM since it converges later
    svm_val.append(cross_val_score(model, X_train, y_train, cv=5).mean())
    model.fit(X_train, y_train)
    svm_train.append(model.score(X_train, y_train))

best_lambda_svm = lambdas[np.argmax(svm_val)]
print("Linear SVM best lambda:", best_lambda_svm)

# train final models and test
logreg = LogisticRegression(C=1/best_lambda_logreg, max_iter=2000)
logreg.fit(X_train, y_train)
print("Logistic regression test accuracy:", logreg.score(X_test, y_test))

svm = LinearSVC(C=1/best_lambda_svm, dual=False, max_iter=5000)
svm.fit(X_train, y_train)
print("Linear SVM test accuracy:", svm.score(X_test, y_test))

# accuracy vs lambda plots
plt.figure()
plt.semilogx(lambdas, logreg_train, "o-", label="Training")
plt.semilogx(lambdas, logreg_val, "s-", label="Cross-validation")
plt.xlabel("lambda")
plt.ylabel("Accuracy")
plt.title("Logistic regression")
plt.legend()
plt.savefig("logreg_curve.png")

plt.figure()
plt.semilogx(lambdas, svm_train, "o-", label="Training")
plt.semilogx(lambdas, svm_val, "s-", label="Cross-validation")
plt.xlabel("lambda")
plt.ylabel("Accuracy")
plt.title("Linear SVM")
plt.legend()
plt.savefig("svm_curve.png")

# confusion matrix for the SVM
names = ["Walking", "Walking upstairs", "Walking downstairs",
         "Sitting", "Standing", "Laying"]
y_pred = svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=names).plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")