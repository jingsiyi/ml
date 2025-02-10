import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve as rc, auc as ac
from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import make_classification as mc
from sklearn.linear_model import LogisticRegression as lr

X, y = mc(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=42)

model = lr()
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = rc(y_test, y_prob)

roc_auc = ac(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
