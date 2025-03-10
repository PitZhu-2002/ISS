import math
def CD(q,k,n):
    return q*math.sqrt(k*(k+1)/(6*n))

print('0.1:',CD(3.0296013125698,12,100))
