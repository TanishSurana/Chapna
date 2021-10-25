'''

    block: 
    loss calculation: 1) pred - out sqaure, 2) ma'am's equation

    2) ma'am eq

    loss == just to print 
    loss > derivative for all independent features
        x1 x2 x3 
        partial derivate -> matrix add = calcuation
        new = old - alpha * partial  

    1) mse
    

'''


m = 1 # these could be random
c = 1 # these could be random

n = len(X_Train)



alpha = 0.05
count = 5000



def derivate_m(Y_Train, Ypredd, X_Train, n):
  return ((-2/n)*(sum(X_Train*(Y_Train - Ypredd))))

def derivate_c(Y_Train, Ypredd, X_Train, n):
  return ((-2/n)*(sum(Y_Train - Ypredd)))


for i in range(count):
  Ypredd = m*X_Train + c
  m = m - alpha*(derivate_m(Y_Train, Ypredd, X_Train, n))
  c = c - alpha*(derivate_c(Y_Train, Ypredd, X_Train, n))