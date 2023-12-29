import numpy as np
import random 
from bitstring import BitArray
max = 2147483647
min = -2147483648

def adder_test_generator():
    for i in range(200):
        a = np.random.randint(min,max)
        b = np.random.randint(min,max)
        cin = np.random.randint(0,2)
        
        C = a + b + cin
        V = 0
        if C > 2147483647:
            V = 1
        if C < -2147483648: 
            V = 1
        print(str(a)+  " " + str(b) + " " + " " + str(cin) + " " + str(C) + " " + str(V))


def leftshifter():

    for i in range(300):
        b = str(f'{random.getrandbits(32):=032b}')
        sa = (np.random.randint(0,32))
        sa_bin = f'{sa:05b}'
        cin = str(np.random.randint(0,2))

        C = b + sa*cin

        C = C[-32:]

        print( b + " " + str(sa_bin) + " "  + cin + " " + C)


for i in range(500):
    op = np.random.randint(0,16)
    a = np.random.randint(min,max)
    b = np.random.randint(min,max)
    
    if a>0:
        a = '{:032b}'.format(a)
    else:
        a = bin(a % (1<<32))[2:]

    if b>0:
        b = '{:032b}'.format(b)
    else:
        b = bin(b % (1<<32))[2:]


    sa = (np.random.randint(0,32))
    sa_bin = '{:05b}'.format(sa)
    C = 0
    V = 0
    if op==0:
        C = BitArray(bin=a).int & BitArray(bin = b).int
        if C>0:
            C = '{:032b}'.format(C)
        else:
            C = bin(C % (1<<32))[2:]
    elif op==1:
        C = BitArray(bin=a).int | BitArray(bin=b).int
        if C>0:
            C = '{:032b}'.format(C)
        else:
            C = bin(C % (1<<32))[2:]
    elif op==10 or op==11:
        C = str(b) + sa*str(0)
        C = C[-32:]
    elif op==13:
        C = BitArray(bin=a).int ^ BitArray(bin=b).int
        if C>0:
            C = '{:032b}'.format(C)
        else:
            C = bin(C % (1<<32))[2:]
    elif op == 15:
        C = BitArray(bin=a).int & BitArray(bin=b).int
        C = ~C
        if C>0:
            C = '{:032b}'.format(C)
        else:
            C = bin(C % (1<<32))[2:]
    elif op ==4 or op==5:
        cin = str(0)
        C = sa*cin + b
        C = C[:32]
    elif op ==6 or op==7:
        MSB=str(b)[0]
        C = sa*MSB + b
        C = C[:32]
    elif op==8:
        C = (BitArray(bin=a).int==BitArray(bin=b).int)
        if C==True:
            C = '{:032b}'.format(0)
        else:
            C = '{:032b}'.format(1)
    elif op==9:
        C = (BitArray(bin=a).int==BitArray(bin=b).int)
        if C==True:
            C = '{:032b}'.format(1)
        else:
            C = '{:032b}'.format(0)
    elif op==14:
        C = BitArray(bin=a).int<0
        if C==True:
            C = '{:032b}'.format(1)
        else:
            C = '{:032b}'.format(0)
    elif op==12:
        C = BitArray(bin=a).int>=0
        if C==True:
            C = '{:032b}'.format(1)
        else:
            C = '{:032b}'.format(0)
    elif op==2:
        C = BitArray(bin=a).int-BitArray(bin=b).int
        if C > 2147483647:
            V = 1
        if C < -2147483648: 
            V = 1
        if C>0:
            C = '{:032b}'.format(C)
        else:
            C = bin(C % (1<<32))[2:]
    elif op ==3:
        C = BitArray(bin=a).int+BitArray(bin=b).int
        if C > 2147483647:
            V = 1
        if C < -2147483648: 
            V = 1
        if C>0:
            C = '{:032b}'.format(C)
        else:
            C = bin(C % (1<<32))[2:]

    if len(C) < 32:
        C = (32 - len(C)) * str(0) + C
    print(str(a)+  " " + str(b) + " " + str(op) + " " +str(sa_bin) + " " + str(C) + " " + str(V))


    




