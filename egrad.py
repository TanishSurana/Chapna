import numpy as np

def partial_derivative(power, coeff, w):
    return power*(w**(power-1))*coeff

def loss(weights, power, coeff, const):
    ws = np.array(weights)
    ps = np.array(power)
    cs = np.array(coeff)

    return round((((ws**ps)*cs).sum() + const), 3)

def main():
    # loss  w1^2 + w2 + c
    alpha = 0.2

    weights = [3, 4]
    power = [2, 2]
    coeff = [1, 1]
    const = 4

    n = len(weights)
    epoch = 5

    # initial loss
    print('initial loss: ', loss(weights, power, coeff, const))
    print('\n\ntraining')



    for pizza in range(epoch):
        # loss, weights 
        print('\n\nepoch: ', pizza)
        for i in range(n):
            pd = partial_derivative(power[i], coeff[i], weights[i])
            weights[i] -= alpha*(pd)
            print(round(pd,3), round(alpha*pd, 3), end='  |  ')
        
        tp = [ round(elem, 3) for elem in weights ]
        print('\nweight: ', tp)
        print('new loss: ', loss(weights, power, coeff, const))


main()