import numpy as np
 
epochs = 20000                                  # Nombre d'itération
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 2, 1
L = .1                                               
 
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0], [1], [1], [0]])
 
def tanh (x):  return 1 / (1 + np.exp(-x))      # fonction d'activation
def tanh_(x): return x * (1 - x)             # dériver fonction d'activation
                                                # poids 
Wh = 0.5
Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))
 
for i in range(epochs):
 
    H = tanh(np.dot(X, Wh))                  # resultat couche cacher
    Z = np.dot(H,Wz)                            # sortir couche
    E = Y - Z                                   # erreur
    dZ = E * L                                  # delta Z
    Wz +=  H.T.dot(dZ)                          # changement des poids de la couche de sortie
    dH = dZ.dot(Wz.T) * tanh_(H)             # delta H
    Wh +=  X.T.dot(dH)                          # changement des poids de la couche cacher
     
print(Z) # resultat apprentissage
print(E)
print("les poids de la sortie",Wz)
