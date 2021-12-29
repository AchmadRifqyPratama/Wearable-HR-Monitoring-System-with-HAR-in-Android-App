import pandas as pd
import numpy as np
import serial
import scipy
from scipy import signal
from scipy import linalg
from pandas import read_csv
from scipy.linalg import hankel
from scipy.sparse.linalg import svds

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None, title=True):
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind, title)

    return ind

def hk(x, fs, window, step):
    L = round(400*100/125)
    N = round(1000*100/125)  
    K=N-L;    
    xc=x[0:K+1]; 
    xr=x[K:N]; 
    H=hankel(xc,xr);
    return H 
    
def auto_grandi(eigen):
    eig = eigen[0:10]     
    dif = -1 * np.diff(eig)  
    media = np.mean(dif) 
    index = np.where(dif>media)
    index = index[0]
    ukuran = len(index)
    if (ukuran == 0):
        ind = 0
    elif (ukuran > 0):
        ind = index[ukuran-1]     
    if np.size(dif) == 0:
        p = 0
        num_aut=9
    else:
        p = dif[ind];         
        if (p > 2*(eigen[0]-eigen[1])):
            num_aut=ind;
        else:
            num_aut=9;
    num_aut=max(4,num_aut+1)
    return num_aut  
    
def kalman_filter(z_ppg1,z_ppg2):
    Q=4**2;                    
    R=10**2;                      
    
    count=0;                           
    countmax=5;
    imax=3;    
    
    xposteriori=(z_ppg1[0]+z_ppg1[0])/2
    res = np.zeros(len(z_ppg1))
    res[0]=(z_ppg1[0]+z_ppg1[0])/2
    Pposteriori=0                           
        
    for i in range (1, len(z_ppg1)):
    
        xpriori=xposteriori;               
        Ppriori=Pposteriori+Q
    
        S=Ppriori+R
        gain=Ppriori/S 
        
        inn1=z_ppg1[i]-xpriori
        inn2=z_ppg2[i]-xpriori
        
        index = np.max([1, i-90])
        VAR1=np.var(z_ppg1[index-1:i])
        VAR2=np.var(z_ppg2[index-1:i])

        
        if(np.isnan(inn1) or np.abs(inn1)> 2*np.sqrt(S)):
            flag1=1
        else:
            flag1=0
        
        if(np.isnan(inn2) or np.abs(inn2)>2*np.sqrt(S)):
            flag2=1
        else:
            flag2=0
        
 
        if (VAR1*2)<VAR2  and i>20 :
          flag2=1
        elif (VAR2*2)<VAR1  and i>20:
            flag1=1

        if((flag1==0 and flag2==0)or((count==countmax or i<imax) and (~np.isnan(inn1) and ~np.isnan(inn2)))):
     
            if(np.abs(inn1)<np.abs(inn2)):
                inn=inn1
            else:
                inn=inn2
         
            xposteriori=xpriori+gain*inn
            Pposteriori=(1-gain)*Ppriori
            if(count==countmax ):
                xposteriori=(z_ppg1[i]+z_ppg2[i]+xpriori)/3
            
            count=0
            
        elif((flag1==0 and flag2==1)or((count==countmax or i<imax) and ~np.isnan(inn1))):                     
            
            inn=inn1
            xposteriori=xpriori+gain*inn
            Pposteriori=(1-gain)*Ppriori
            if(count==countmax ):
                xposteriori=0.5*(z_ppg1[i]+xpriori)
            count=0
            
        elif((flag2==0 and flag1==1) or ((count==countmax or i<imax) and ~np.isnan(inn2))) :                    
     
            inn=inn2;
            xposteriori=xpriori+gain*inn
            Pposteriori=(1-gain)*Ppriori
            if(count==countmax ):
                xposteriori=0.5*(z_ppg2[i]+xpriori)
            
            count=0
        else:
            count=count+1
            xposteriori=xpriori
            Pposteriori=Ppriori
            
        res[i]=xposteriori
        
    return xposteriori
  
def main(fin, ind, i, PPG1, PPG2, heartbeat, AX, AY, AZ):
	fs = 100
	window = 8
	step = 1
	N=len(PPG1)
	n=len(PPG1)
	ppgxx=PPG1[ind:fin]
	ppgyy=PPG2[ind:fin]
#
#     #ppgxx = scipy.signal.detrend(ppgx, type = 'linear')*10
# 	#ppgyy = scipy.signal.detrend(ppgy, type = 'linear')*10
#
	accx=AX[ind:fin]
	accy=AY[ind:fin]
	accz=AZ[ind:fin]

    #PPG1 = np.array(PPG1)
	#ppgxx=pd.DataFrame(list(PPG1)).T
	#ppgyy=pd.DataFrame(list(PPG2)).T
	#accx= pd.DataFrame(list(AX)).T
	#accy= pd.DataFrame(list(AY)).T
	#accz= pd.DataFrame(list(AZ)).T


	H1 = hk(ppgxx, fs, window, step)
	U1, S1, V1 = np.linalg.svd(H1, full_matrices = False)
	eig1 = np.diag(S1)

	H2 = hk(ppgyy, fs, window, step)
	U2, S2, V2 = np.linalg.svd(H2, full_matrices = False)
	eig2 = np.diag(S2)

	H_x = hk(accx, fs, window, step)
	U_x, S_x, V_x = np.linalg.svd(H_x, full_matrices = False)
	eig_x = np.diag(S_x)

	H_y = hk(accy, fs, window, step)
	U_y, S_y, V_y = np.linalg.svd(H_y, full_matrices = False)
	eig_y = np.diag(S_y)

	H_z = hk(accz, fs, window, step)
	U_z, S_z, V_z = np.linalg.svd(H_z, full_matrices = False)
	eig_z = np.diag(S_z)

	ppg_aut_1=auto_grandi(S1)
	ppg_aut_2=auto_grandi(S2)
	acc_aut_x=auto_grandi(S_x)
	acc_aut_y=auto_grandi(S_y)
	acc_aut_z=auto_grandi(S_z)

	Matrix_Cor_1x=np.matmul((U1.T),(U_x))
	Matrix_Cor_1y=np.matmul((U1.T),(U_y))
	Matrix_Cor_1z=np.matmul((U1.T),(U_z))
	Matrix_Cor_2x=np.matmul((U2.T),(U_x))
	Matrix_Cor_2y=np.matmul((U2.T),(U_y))
	Matrix_Cor_2z=np.matmul((U2.T),(U_z))

	Matrix_Cor_1x=Matrix_Cor_1x[0:ppg_aut_1,0:acc_aut_x]
	Matrix_Cor_1y=Matrix_Cor_1y[0:ppg_aut_1,0:acc_aut_y]
	Matrix_Cor_1z=Matrix_Cor_1z[0:ppg_aut_1,0:acc_aut_z]
	Matrix_Cor_2x=Matrix_Cor_2x[0:ppg_aut_2,0:acc_aut_x]
	Matrix_Cor_2y=Matrix_Cor_2y[0:ppg_aut_2,0:acc_aut_y]
	Matrix_Cor_2z=Matrix_Cor_2z[0:ppg_aut_2,0:acc_aut_z]

	Matrix_Cor_1x=np.multiply(Matrix_Cor_1x, Matrix_Cor_1x)
	Matrix_Cor_1y=np.multiply(Matrix_Cor_1y, Matrix_Cor_1y)
	Matrix_Cor_1z=np.multiply(Matrix_Cor_1z, Matrix_Cor_1z)
	Matrix_Cor_2x=np.multiply(Matrix_Cor_2x, Matrix_Cor_2x)
	Matrix_Cor_2y=np.multiply(Matrix_Cor_2y, Matrix_Cor_2y)
	Matrix_Cor_2z=np.multiply(Matrix_Cor_2z, Matrix_Cor_2z)

	Matrix_Cor_1x=(Matrix_Cor_1x.T)
	Matrix_Cor_1y=(Matrix_Cor_1y.T)
	Matrix_Cor_1z=(Matrix_Cor_1z.T)
	Matrix_Cor_2x=(Matrix_Cor_2x.T)
	Matrix_Cor_2y=(Matrix_Cor_2y.T)
	Matrix_Cor_2z=(Matrix_Cor_2z.T)

	SUM_1x= [max(idx) for idx in zip(*Matrix_Cor_1x)]
	SUM_1y =[max(idx) for idx in zip(*Matrix_Cor_1y)]
	SUM_1z= [max(idx) for idx in zip(*Matrix_Cor_1z)]
	SUM_2x= [max(idx) for idx in zip(*Matrix_Cor_2x)]
	SUM_2y= [max(idx) for idx in zip(*Matrix_Cor_2y)]
	SUM_2z= [max(idx) for idx in zip(*Matrix_Cor_2z)]

	SUM_PART_1 = [SUM_1x, SUM_1y, SUM_1z]
	SUM_PART_2 = [SUM_2x, SUM_2y, SUM_2z]

	SUM_1 = [max(idx) for idx in zip(*SUM_PART_1)]
	SUM_2 = [max(idx) for idx in zip(*SUM_PART_2)]

	SOGLIA = 0.6
	Sr_1 = np.zeros((len(S1), len(S1)))
	Sr_2 = np.zeros((len(S2), len(S2)))
	cont = 0
	for x in range (0, len(SUM_1)):
		if SUM_1[x]<SOGLIA:
			Sr_1[x,x]=S1[x]
			cont=cont+1

	if cont==0 :
		for x in (0, len(SUM_1)):
			Sr_1[x,x]=S1[x]

	temp_1 = np.zeros((ppg_aut_1, len(U1)))
	temp_mH_1 = np.matmul(U1, Sr_1)
	mH_1 = np.matmul(temp_mH_1, V1)

	for x in range (0, ppg_aut_1):
		temp_1[x,:] = mH_1[:,x]/np.sqrt(S1[x])
	if ppg_aut_1 >1:
		xr_1=temp_1.sum(axis = 0)
	else:
		xr_1=temp_1


	cont = 0
	for x in range(0, len(SUM_2)):
		if SUM_2[x]<SOGLIA:
			Sr_2[x,x]=S2[x]
			cont=cont+1

	if cont==0 :
		for x in range (0, len(SUM_2)):
			Sr_2[x,x]=S2[x]

	temp_2 = np.zeros((ppg_aut_2, len(U2)))
	temp_mH_2 = np.matmul(U2, Sr_2)
	mH_2 = np.matmul(temp_mH_2, V2)

	for x in range(0,ppg_aut_2):
		temp_2[x,:] = mH_2[:,x]/np.sqrt(S2[x])
	if ppg_aut_2 >1:
		xr_2=temp_2.sum(axis = 0)
	else:
		xr_2=temp_2

	temp1 = xr_1
	temp2 = xr_2
	fS=fs
	Nfft=1024
	F=fS/Nfft
	w_old=np.hanning(len(xr_1))
	w = w_old.T
	w2_old=np.hanning(len(xr_2))
	w2 = w2_old.T
	temp_Xr1 = np.multiply(xr_1, w)
	temp_Xr2 = np.multiply(xr_2, w2)
	Xr_1 = np.abs(np.fft.fft(temp_Xr1, n = Nfft)/np.sum(w))
	Xr_2 = np.abs(np.fft.fft(temp_Xr2, n = Nfft)/np.sum(w2))

	start = np.round((60/60)/F)
	fine = np.round((180/60)/F)

	Xr1_temp = Xr_1[int(start)-1:int(fine)]
	results1 = detect_peaks(Xr1_temp)
	peak_matrix1 = np.zeros((len(results1), 2))
	for x in range (0, len(results1)):
		peak_matrix1[x, 0] = Xr1_temp[results1[x]]
		peak_matrix1[x, 1] = results1[x]
	peak_matrix1 = peak_matrix1[peak_matrix1[:, 0].argsort()]

	Xr2_temp = Xr_2[int(start)-1:int(fine)]
	results2 = detect_peaks(Xr2_temp)
	peak_matrix2 = np.zeros((len(results2), 2))
	for x in range (0, len(results2)):
		peak_matrix2[x, 0] = Xr2_temp[results2[x]]
		peak_matrix2[x, 1] = results2[x]
	peak_matrix2 = peak_matrix2[peak_matrix2[:, 0].argsort()]


	ind_pk_1 = np.flip(peak_matrix1[:,1])
	ind_pk_2 = np.flip(peak_matrix2[:,1])

	w_old = np.hanning(len(ppgxx))
	w = w_old.T
	w2_old =np.hanning(len(ppgyy));
	w2 = w2_old.T


	temp_X1 = np.multiply(ppgxx, w)
	temp_X2 = np.multiply(ppgyy, w2)

	X1 = np.abs(np.fft.fft(temp_X1, n = Nfft)/np.sum(w))
	X2 = np.abs(np.fft.fft(temp_X2, n = Nfft)/np.sum(w2))

	X1_temp = X1[int(start)-1:int(fine)]
	results_X1 = detect_peaks(X1_temp)
	peak_matrix_X1 = np.zeros((len(results_X1), 2))
	for x in range (0, len(results_X1)):
		peak_matrix_X1[x, 0] = X1_temp[results_X1[x]]
		peak_matrix_X1[x, 1] = results_X1[x]
	peak_matrix_X1 = peak_matrix_X1[peak_matrix_X1[:, 0].argsort()]

	X2_temp = X2[int(start)-1:int(fine)]
	results_X2 = detect_peaks(X2_temp)
	peak_matrix_X2 = np.zeros((len(results_X2), 2))
	for x in range (0, len(results_X2)):
		peak_matrix_X2[x, 0] = X2_temp[results_X2[x]]
		peak_matrix_X2[x, 1] = results_X2[x]
	peak_matrix_X2 = peak_matrix_X2[peak_matrix_X2[:, 0].argsort()]


	ind1 = np.flip(peak_matrix_X1[:,1])
	ind2 = np.flip(peak_matrix_X2[:,1])

	if (np.size(ind1)==0 and np.size(ind_pk_1) == 0):
		p1=heartbeat
	elif (np.size(ind1) == 0):
		p1=(ind_pk_1[0]+start-1)*F*60
	elif (np.size(ind_pk_1) > 0):
		p1=(ind_pk_1[0]+start-1)*F*60
		picco1=(ind1[0]+start-1)*F*60
		if np.abs(2*picco1-p1)<10 :
			p1=picco1
	else:
		p1=heartbeat


	if (np.size(ind2)==0 and np.size(ind_pk_2) == 0):
		p2=heartbeat
	elif (np.size(ind2) == 0):
		p2=(ind_pk_2[0]+start-1)*F*60
	elif (np.size(ind_pk_2) > 0):
		p2=(ind_pk_2[0]+start-1)*F*60
		picco2=(ind2[0]+start-1)*F*60
		if np.abs(2*picco2-p2)<10 :
			p2=picco2
	else:
		p2 = heartbeat
		
	return round((p1+p2)/2)
