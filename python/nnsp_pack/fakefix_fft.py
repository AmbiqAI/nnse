import numpy as np
from .converter_fix_point import fakefix, cplx_fakefix, float2hex
is_fixPt = 1
eNt = 8
Nt = 2**eNt # table size. Support upto Nt-pt fft
Ntr = Nt * 2
def bitreverse(bits):
    
    y = np.zeros(2**bits)

    for m in range(2**bits):
        x = m
        for i in range(bits):
            y[m] = y[m] * 2 + (x % 2)
            x = (x - (x % 2)) / 2

    return y   

def fft(x, make_c_table=False):
    Qbit = 15
    if len(x)==128:
        eN = 7
    elif len(x)==256:
        eN = 8
    N = 2**eN
    r = np.log2(N) % 2
    stages = int((np.log2(N) - r) / np.log2(4))
   
    brt = bitreverse(int(np.log2(Nt))) # table to save
   
    R = 2**(eNt-eN)
    br = (brt[:int(Nt/R)] / R).astype(np.int32); 
 
    seq =np.arange(Nt >> 2)
    tw = np.exp(-2*1j * np.pi / Nt * seq)
    TW = np.stack([tw**0, tw**2, tw**1, tw**3]).T # % table to save
    TW_r = np.real(TW)
    TW_i = np.imag(TW)
    if is_fixPt :   
        TW_r = fakefix(TW_r, 16,Qbit)
        TW_i = fakefix(TW_i, 16,Qbit)
    
    TW = TW_r + 1j * TW_i

    if make_c_table:
        file = open('../lib_nnsp/src/twiddle_fft_dif.c', 'w')
        file.write('// Twiddle factor with %d-point FFT\n' % 512)
        file.write('// exp(j2*pi/Nt*2 * k) exp(j2*pi/Nt * k) exp(j2*pi/Nt * 3 * k), k in 0~Nt-1\n')
        file.write('// where and k in 0~Nt/4-1\n')
        file.write('// Each complex value is of 32bit with each real and imag 16bit, respectively\n\n')


        file.write('#include <stdint.h>\n')

        file.write('const int32_t fft_tw_coeff[] = {\n')
        for i in range(Nt >> 2):
        
            file.write('0x%04hx%04hx, 0x%04hx%04hx, 0x%04hx%04hx, 0x%04hx%04hx,\n' %
                                        (   float2hex(TW_i[i,0]), 
                                            float2hex(TW_r[i,0]), 
                                            float2hex(TW_i[i,1]), 
                                            float2hex(TW_r[i,1]), 
                                            float2hex(TW_i[i,2]), 
                                            float2hex(TW_r[i,2]), 
                                            float2hex(TW_i[i,3]), 
                                            float2hex(TW_r[i,3]))  
                                        )
            
        
        file.write('};\n')

    #####################
    Ntr = Nt * 2
    seq = np.arange(Ntr >> 1)
    rtw = np.exp(-2* 1j * np.pi / Ntr * seq)

    rTW_r = np.real(rtw)
    rTW_i = np.imag(rtw)
    if is_fixPt:    
        rTW_r = fakefix(rTW_r, 16,Qbit)
        rTW_i = fakefix(rTW_i, 16,Qbit)
    rTW = rTW_r + 1j * rTW_i
    if make_c_table:
        file.write('const int32_t rfft_tw_coeff[] = {\n')

        for i in range(Ntr >> 1):  
            file.write('0x%04hx%04hx, ' % 
                                        (float2hex(rTW_i[i]), 
                                        float2hex(rTW_r[i])) 
                                        )
            

        file.write('};\n')
        #####################################################3


        file.write('const int16_t br_coeff[] = {\n')
        
        for i in range(Nt):
            file.write('%d, ' % brt[i])
            
        file.write('};\n')
        file.close()
    y = x.copy()
    # print(y.shape)
    M = [[1,  1,   1,   1],
         [1,  1,  -1,  -1],
         [1, -1, -1j,  1j],
         [1, -1,  1j, -1j]]
    M = np.array(M)
    Nf = N
    Ng = 1
    
    S = 2**(eNt-eN)

    t = np.zeros(4, dtype=np.complex64)

    for s in range(stages):
        Nfd4 = Nf >> 2
        for g in range(Ng):
            k = 0
            s_idx = g * Nf
            for m in range(Nfd4):
                idx = s_idx                
                t[0] = y[idx]
                idx = idx + Nfd4
                t[2] = y[idx]
                idx = idx + Nfd4
                t[1] = y[idx]
                idx = idx + Nfd4
                t[3] = y[idx]

                # t = (M*t) .* TW(k,:).';
                if is_fixPt:  
                    Mt = cplx_fakefix( np.matmul(M,t), 32, 15)
                    t = cplx_fakefix(Mt * TW[k,:], 32, 15)
                else:
                    t = np.matmul(M, t) * TW[k,:]
                    
                k = k+S
                idx = s_idx
                y[idx] = t[0]
                idx = idx + Nfd4
                y[idx] = t[1]
                idx = idx + Nfd4
                y[idx] = t[2]
                idx = idx + Nfd4
                y[idx] = t[3]
            
                s_idx = s_idx+1
            

        Nf = Nf >> 2
        Ng = Ng << 2
        S  = S << 2

    if r == 1:
        
        t = np.zeros(2, dtype=np.complex64)
        M = [[1,  1],
             [1, -1]]
        M = np.array(M)
        Ng = N >> 1
        idx = np.arange(2)
        for i in range(Ng):
            t = np.matmul(M, y[idx])
            
            if is_fixPt:  
                t = cplx_fakefix(t, 32, 15)
            
            y[idx] = t
            idx = idx+2
    y = y[br]
    if  0:
        for i in range(N):
            print('%d: (%d, %d)\n' % (i, np.real(y[i]) * 2**15, np.imag(y[i]) * 2**15), end='')
    yref = np.fft.fft(x)
   
    # print(abs(y - yref).max())
    return y

def rfft(xr_, num_rfft, make_c_table=False):
    if num_rfft==256:
        R=2
    elif num_rfft==512:
        R=1
    xr=np.zeros(num_rfft, dtype=np.complex64)
    xr[:len(xr_)] = xr_
    N = len(xr) >> 1
    x = xr[::2] + 1j * xr[1::2]
    y = fft(x,make_c_table=make_c_table)
    
    #  cmplx to real
    Nr = N << 1
    Xe = np.zeros(Nr >> 1, dtype=np.complex64)
    Xo = np.zeros(Nr >> 1, dtype=np.complex64)

    tmp = np.conj(y[0])
    Xe[0] = (y[0] + tmp) / 2.0
    Xo[0] = -1j * (y[0] - tmp) / 2.0
    if is_fixPt:  
        Xe[0] = cplx_fakefix(Xe[0], 32, 15)
        Xo[0] = cplx_fakefix(Xo[0], 32, 15)
    
    for i in range(1, Nr >> 1):
        idx = (Nr >> 1) - i
        tmp = np.conj(y[idx])
        Xe[i] = (y[i] + tmp) / 2.0
        Xo[i] = -1j * (y[i] - tmp) / 2.0

        if is_fixPt:  
            Xe[i] = cplx_fakefix(Xe[i], 32, 15)
            Xo[i] = cplx_fakefix(Xo[i], 32, 15)
      
    # for i = 1 : Nr/2
    #     fprintf('%d: (%d, %d)\n', i-1, real(Xo(i)) * 2^15, imag(Xo(i)) * 2^15)
    # end
    k = np.arange(Ntr >> 1)
    
    tw = np.exp(-1j * 2 * np.pi / Ntr * k)


    if is_fixPt:
        tw = cplx_fakefix(tw, 16, 15)
    
    X = np.zeros(1 + (Nr >> 1), dtype=np.complex64)
    # X = Xe + Xo .* tw
    for i in range(Nr >> 1):
        X[i] = Xe[i] + cplx_fakefix(Xo[i] * tw[R*i], 32, 15)
    X[Nr>>1] = Xe[0] - Xo[0]
    if is_fixPt:
        X = cplx_fakefix(X, 32, 15)
    
    # for i in range(1 + (Nr >> 1)):
    #     print('%d: (%d, %d)\n' % (i, np.real(X[i]) * 2**15, np.imag(X[i]) * 2**15), end='')
    

    Xref = np.fft.fft(xr)
    
    Xref = Xref[: (Nr>>1)+1]
    # print(np.abs(Xref-X).max())
    return X
if __name__=="__main__":
    MAKE_DATA = False
    N = 512
    
    if MAKE_DATA:
        xr = fakefix(100 * np.random.randn(N) / 2**15, 32, 15) 
        file = open('dump/fft_in', 'w')
        for i in range(N):
            file.write('%d ' % (xr[i] * 2**15))  
        file.close()

        X = rfft(xr, N)
        file = open('dump/fft_out_py', 'w')
        for i in range(1 + (N >> 1)):
            file.write('%d %d\n' % (X[i].real * 2**15, X[i].imag * 2**15))  
        file.close()
    else:
        py = np.loadtxt("dump/fft_out_py",dtype =int)
        cc = np.loadtxt("dump/fft_out_c",dtype =int)
        e = py-cc
        print(e.max())