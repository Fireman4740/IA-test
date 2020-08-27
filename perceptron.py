from random import choice
import numpy 

#fonction de seuil 
unit_step = lambda x: 0 if 1 / (1 + numpy.exp(-x)) < 0 else 1

#base de données de tradning 
training_data = [
    (array([13,1,2]), 0), #1
    (array([6,2,2]), 0),
    (array([6,3,2]), 0),
    (array([20,4,2]), 0),
    (array([16,1,3]), 0),#5
    (array([5,3,3]), 0),
    (array([11,4,3]), 0),
    (array([4,5,3]), 0),
    (array([16,5,3]), 0),
    (array([2,1,4]), 0),#10
    (array([20,1,4]), 0),
    (array([1,2,4]), 0),
    (array([5,2,4]), 0),
    (array([6,2,4]), 0),
    (array([13,2,4]), 0),#15
    (array([16,3,4]), 0),
    (array([3,4,4]), 0),
    (array([6,5,4]), 0),
    (array([7,5,4]), 0),
    (array([9,1,5]), 0),#20
    (array([2,3,5]), 0),
    (array([13,3,5]), 0),
    (array([20,3,5]), 0),
    (array([7,4,5]), 0),
    (array([10,4,5]), 0),#25
    (array([13,4,5]), 0),
    (array([15,4,5]), 0),
    (array([19,4,5]), 0),
    (array([9,2,6]), 0),
    (array([18,2,6]), 0),#30
    (array([2,3,6]), 0),
    (array([4,5,6]), 0),
    (array([20,5,6]), 0),#
    (array([14,1,7]), 0),
    (array([7,2,7]), 0),#35
    (array([12,3,7]), 0),
    (array([15,3,7]), 0),
    (array([12,5,7]), 0),
    (array([2,1,8]), 0),
    (array([15,1,8]), 0),#40
    (array([18,1,8]), 0),
    (array([2,3,8]), 0),
    (array([2,4,8]), 0),
    (array([10,4,8]), 0),
    (array([3,5,8]), 0),#45
    (array([12,5,8]), 0),
    (array([8,1,9]), 0),
    (array([3,2,9]), 0),
    (array([10,2,9]), 0),
    (array([15,3,9]), 0),#50
    (array([6,4,9]), 0),
    (array([8,4,9]), 0),
    (array([11,4,9]), 0),
    (array([12,5,9]), 0),
    (array([17,2,10]), 0),#55
    (array([8,4,10]), 0),
    (array([18,4,10]), 0),
    (array([17,5,10]), 0),
    (array([1,2,11]), 0),
    (array([10,2,11]), 0),#60
    (array([14,3,11]), 0),
    (array([14,3,11]), 0),
    (array([20,3,11]), 0),
    (array([15,5,11]), 0),
    (array([5,1,12]), 0),#65
    (array([11,1,12]), 0),
    (array([14,1,12]), 0),
    (array([15,1,12]), 0),
    (array([9,2,12]), 0),
    (array([10,2,12]), 0),#70
    (array([12,3,12]), 0),
    (array([18,3,12]), 0),
    (array([6,4,12]), 0),
    (array([18,5,12]), 0),
    (array([7,1,13]), 0),#75
    (array([19,1,13]), 0),
    (array([12,2,13]), 0),
    (array([20,2,13]), 0),
    (array([14,3,13]), 0),
    (array([11,4,13]), 0),#80
    (array([14,4,13]), 0),
    (array([18,4,13]), 0),
    (array([2,1,14]), 0),
    (array([13,2,14]), 0),
    (array([1,2,1]), 0),#85
    (array([8,5,1]), 0),###############
    (array([19,2,2]), 1),
    (array([6,2,4]), 1),
    (array([11,1,6]), 1),
    (array([12,2,7]), 1),#90
    (array([6,3,9]), 1),
    (array([7,2,10]), 1),
    (array([18,2,10]), 1),
    (array([5,4,13]), 1),
    (array([14,2,14]), 1),#95
    (array([7,4,14]), 1),
    (array([9,4,14]), 1),
    (array([9,4,14]), 1),
    (array([3,8,1]), 1),
    (array([8,3,1]), 1),#100
]
#numéro de l'itération
ni=1
#initialisation des poids w
w = random.rand(3)
#variable de l'erreur
errors = []
#coefficient d'apprantissage
eta = 1/ni
#nombre d'itération
n = 10000

for i in range(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x
    ni+=1
    print(w,"--",ni,"--",error,"--",)
    
    


for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

print(error)
print(w)

