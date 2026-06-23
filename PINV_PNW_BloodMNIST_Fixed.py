
import numpy as np
import cv2
from medmnist import BloodMNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import mode

# ---------------- Feature Extraction ---------------- #

def original(img):
    return img.flatten()/255.0

def histogram(img):
    return cv2.equalizeHist(img).flatten()/255.0

def threshold(img):
    _, th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return th.flatten()/255.0

def sobel(img):
    g = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
    g = np.abs(g)
    g = g/(g.max()+1e-8)
    return g.flatten()

# ---------------- PINV Classifier ---------------- #

class PINVClassifier:

    def __init__(self,input_dim,hidden=200,seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((input_dim,hidden))
        self.B = rng.standard_normal(hidden)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def fit(self,X,Y):
        H = self.sigmoid(X@self.W + self.B)
        classes = int(np.max(Y))+1
        T = np.eye(classes)[Y]
        self.Beta = np.linalg.pinv(H) @ T

    def predict(self,X):
        H = self.sigmoid(X@self.W + self.B)
        O = H @ self.Beta
        return np.argmax(O,axis=1)

# ---------------- Dataset ---------------- #

def load_split(split):

    ds = BloodMNIST(split=split, download=True)

    feats=[[],[],[],[]]
    labels=[]

    for img,label in ds:

        img=np.array(img)

        if img.ndim==3:
            img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        label=int(np.squeeze(label))

        feats[0].append(original(img))
        feats[1].append(histogram(img))
        feats[2].append(threshold(img))
        feats[3].append(sobel(img))

        labels.append(label)

    feats=[np.asarray(f,dtype=np.float32) for f in feats]
    labels=np.asarray(labels,dtype=np.int64)

    return feats,labels

# ---------------- Main ---------------- #

print("Loading BloodMNIST...")
Xtr,Ytr=load_split("train")
Xte,Yte=load_split("test")

names=["Original","Histogram","Threshold","Sobel"]
models=[]

for i in range(4):
    print(f"\nTraining {names[i]}...")
    model=PINVClassifier(Xtr[i].shape[1],hidden=200,seed=i+1)
    model.fit(Xtr[i],Ytr)
    pred=model.predict(Xtr[i])
    print(f"Training Accuracy : {accuracy_score(Ytr,pred)*100:.2f}%")
    models.append(model)

votes=[]
for i,m in enumerate(models):
    votes.append(m.predict(Xte[i]))

votes=np.asarray(votes)
final_pred=mode(votes,axis=0,keepdims=False).mode

print("\n========== FINAL RESULTS ==========")
print(f"Accuracy : {accuracy_score(Yte,final_pred)*100:.2f}%")
print(f"Precision: {precision_score(Yte,final_pred,average='macro',zero_division=0):.4f}")
print(f"Recall   : {recall_score(Yte,final_pred,average='macro',zero_division=0):.4f}")
print(f"F1 Score : {f1_score(Yte,final_pred,average='macro',zero_division=0):.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(Yte,final_pred))
