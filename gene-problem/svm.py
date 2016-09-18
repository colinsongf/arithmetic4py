from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
def analysis(X=[[0,0,0],[1,1,7],[2,3,6]],Y=[0,1,1]):
    #clf=svm.NuSVC()
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
    #X = np.random.randn(300, 2)
    X=np.asarray(X)
    #Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    Y=np.asarray(Y)
    print(type(X),type(Y))

    clf=svm.SVC()
    clf.fit(X,Y)
    """
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    """
    Z=clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.PuOr_r)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                           linetypes='--')
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired)
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.show()

if __name__=="__main__":
    analysis()