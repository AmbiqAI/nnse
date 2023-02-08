from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plt
from .converter_fix_point import fakefix

log_coeff = [0x0000, 0x7fff, 0x00ff, 0x7f01, 0x01fc, 0x7e07, 0x02f7, 0x7d11, 0x03f0, 0x7c1f, 0x04e7, 0x7b30, 0x05dd, 0x7a44, 0x06d0, 0x795c, 0x07c2, 0x7878, 0x08b2, 0x7797, 0x09a0, 0x76b9, 0x0a8d, 0x75de, 0x0b78, 0x7507, 0x0c61, 0x7432, 0x0d49, 0x7361, 0x0e2f, 0x7292, 0x0f13, 0x71c7, 0x0ff6, 0x70fe, 0x10d7, 0x7038, 0x11b7, 0x6f74, 0x1295, 0x6eb3, 0x1371, 0x6df5, 0x144d, 0x6d3a, 0x1526, 0x6c80, 0x15ff, 0x6bca, 0x16d6, 0x6b15,
0x17ab, 0x6a63, 0x187f, 0x69b4, 0x1952, 0x6906, 0x1a23, 0x685b, 0x1af3, 0x67b2, 0x1bc2, 0x670b, 0x1c8f, 0x6666, 0x1d5c, 0x65c3, 0x1e27, 0x6522, 0x1ef0, 0x6483, 0x1fb9, 0x63e7, 0x2080, 0x634c, 0x2146, 0x62b2, 0x220b, 0x621b, 0x22ce, 0x6186, 0x2391, 0x60f2, 0x2452, 0x6060, 0x2512, 0x5fd0, 0x25d1, 0x5f41, 0x268f, 0x5eb4, 0x274c, 0x5e29, 0x2808, 0x5d9f, 0x28c3, 0x5d17, 0x297c, 0x5c90, 0x2a35, 0x5c0b, 0x2aec, 0x5b87, 0x2ba3, 0x5b05, 0x2c59, 0x5a84, 0x2d0d, 0x5a05, 0x2dc1, 0x5987, 0x2e73, 0x590b, 0x2f25, 0x588f, 0x2fd5, 0x5816, 0x3085, 0x579d, 0x3134, 0x5726, 0x31e2, 0x56b0, 0x328f, 0x563b, 0x333b, 0x55c7, 0x33e6, 0x5555, 0x3490, 0x54e4, 0x3539, 0x5474, 0x35e2, 0x5405, 0x3689, 0x5397, 0x3730, 0x532a, 0x37d6, 0x52bf, 0x387b, 0x5254, 0x391f, 0x51eb, 0x39c3, 0x5183, 0x3a65, 0x511b, 0x3b07, 0x50b5, 0x3ba8, 0x5050, 0x3c49, 0x4fec, 0x3ce8, 0x4f88, 0x3d87, 0x4f26, 0x3e25, 0x4ec4, 0x3ec2, 0x4e64, 0x3f5e, 0x4e04, 0x3ffa, 0x4da6, 0x4095, 0x4d48, 0x412f, 0x4ceb, 0x41c8, 0x4c8f, 0x4261, 0x4c34, 0x42f9, 0x4bda, 0x4391, 0x4b80, 0x4427, 0x4b27, 0x44bd, 0x4ad0, 0x4553, 0x4a79, 0x45e7, 0x4a22, 0x467b, 0x49cd, 0x470e,
0x4978, 0x47a1, 0x4924, 0x4833, 0x48d1, 0x48c4, 0x487e, 0x4955, 0x482d, 0x49e5, 0x47dc, 0x4a74, 0x478b, 0x4b03, 0x473c, 0x4b91, 0x46ed, 0x4c1f, 0x469e, 0x4cac, 0x4651, 0x4d38, 0x4604, 0x4dc4, 0x45b8, 0x4e4f, 0x456c, 0x4eda, 0x4521, 0x4f64, 0x44d7, 0x4fed, 0x448d, 0x5076, 0x4444, 0x50fe, 0x43fb, 0x5186, 0x43b3, 0x520d, 0x436c, 0x5293, 0x4325, 0x5319, 0x42df, 0x539f, 0x429a, 0x5424, 0x4254, 0x54a8, 0x4210, 0x552c, 0x41cc, 0x55af, 0x4189, 0x5632, 0x4146, 0x56b5, 0x4104, 0x5736, 0x40c2, 0x57b8, 0x4081, 0x5838, 0x4040 ]

# /*******************************************************
#     norm_oneTwo normalizes input to [1, 2)

#     input: x in Q15
#     output: 
#             y       : in [1,2) in Q15
#             shift   : integer
#     x = y * 2^shift
# ********************************************************/
def norm_oneTwo(x):
    one = 1
    shift = 0
    for i in range(31):
        idx = one << (30 - i)
        if (idx & x):
            shift = -(30 - i - 15)
            break
    if shift > 0:
        y = x << shift
    else:
        y = x >> -shift
    shift = -shift
    return y, shift


# /*******************************************************
#     my_log is fixed point of natural log function
#     input: x in Q15
#     output: y = log(x) in Q15
    
#     Algorithm: 
#         x = a * 2^k, where
#         a in [1, 2), k is integer
#         y = log(x) = log(a) + k * log(2)

# ********************************************************/
def log10_fix(x):
    LOG2_DIV_LOG10_Q15 = 9864 # log(2) / log(10)
    INV_LOG10_Q15 = 14230 # 1/log(10)
    s = 1 << 15 #1 in Q15
    if (x==0):
        x=1

    
    y, shfit = norm_oneTwo(x)
    kx = (y - s) >> 8
    dx = (y - s) - (kx << 8)
    out = log_coeff[kx << 1] + ((log_coeff[1 + (kx << 1)] * dx) >> 15)
    out = (out * INV_LOG10_Q15) >> 15
    out += (LOG2_DIV_LOG10_Q15 * shfit)
    return out

def log10_fakeFix(xv):
    y = xv.copy()
    for i, x in enumerate(xv):
        xn = int(x*2**15)
        yn = log10_fix(xn)
        y[i]  = yn / 2**15
    return y
if __name__ == '__main__':
    MAKE_DATA = False
    if MAKE_DATA:
        x = np.linspace(0.001, 50, 1000)
        x = fakefix(x, 32, 15)
        file = open('dump/log_in', 'w')
        for xx in x:
            file.write('%d ' % (xx*2**15))
        file.close()
  
        y = log10_fakeFix(x)
        file = open('dump/log_out_py', 'w')
        for yy in y:
            file.write('%d \n' % (yy * 2**15))
        file.close()
        
        yr = np.log10(x)
        plt.figure(1)
        plt.plot(x,y)
        plt.plot(x,yr)
        err = np.abs(y-yr).max()
        print(err)
    else:
        cc = np.loadtxt('dump/log_out_c', dtype=int)
        py = np.loadtxt('dump/log_out_py', dtype=int)
        print(np.abs(cc-py).max())
        plt.figure(1)
        plt.plot(cc)
        plt.plot(py)


    
    

    plt.show()