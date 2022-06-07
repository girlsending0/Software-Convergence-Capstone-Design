#### 2022-2 Software Convergence Capstone Design
# VR 환경에서 EEG 신호를 이용한 감정인식 알고리즘 개발

> 참여: 조혜정, 
> 지도교수: 이원희

# Overview
> ### Introduction
 * 감정은 인간의 인지에 필수적인 역할을 하며 AI가 인간의 감정을 인지하고 이해할 수 있는 능력을 가진다면 인간의 삶을 보다 편리하게 만들 수 있다. 예를 들어 HCI(인간-컴퓨터 상호작용 기술)을 활용해 의식 장애 환자의 감정 상태를 확인할 수 있으며 VR 기반 생체 신호 측정을 통해 사용자의 감정을 식별할 수 있다. 따라서 감정연구를 통해 기업의 시장 조사 등 여러 분야에 적용이 가능하다.   
 * 감정에 관한 연구를 위해 Eye-tracking, ECG, EEG, HRV ,GSR 등과 같은 여러 생리적 신호가 수집되어 왔다. 이 중 EEG(뇌 전도) 데이터는 실험자가 직접 제어하거나 의도할 수 없고 중추 신경계의 전기적 활동을 효과적으로 반영할 수 있다. 이전의 감정 인식 EEG 데이터의 MIPs(기분 유도 절차)는 대부분 2D(ex. 영상물 시청)상황에서 수집되었다. 하지만 2D 디스플레이를 통한 감정 유도는 몰입감과 존재감이 부족하여 3D(실제 상황)에 적용하기 부족할 수 있다. 데이터가 2D가 아닌 3D 몰입형 VR환경에서 수집된다면 현실 세계와 분리되고 가상세계에 몰입하여 보다 현실 상황에 가까운 감정 유도가 가능하다.  
> ### VREED Data  
 * VREED[1]는 3D 몰입형 VR상황에서 MIPs(기분 유도 절차)가 이루어진 59채널 고밀도 EEG 데이터 세트이다.
 * 자극 : 상하이 영화 아카데미에서 제작한 60개의 3D VR 비디오, 상하이 랜드마크(동방명주, 와이탄), 거리풍경, 학교 축제에서 촬영, 비디오는 4초 길이, 4096x2048의 해상도와 초당 30프레임 속도로 H.264형식으로 인코딩 되었다.
* 이러한 자극을 SAM (Self-Assessment-Manikin) 설문을 통해 Positive/Neutral/Negative 범주로 나누었다.

> ### Project Goal
 * VREED EEG signal의 Connectivity Network Analysis 기법을 통해 감정(Positive/Neutral/Negative)에 따른 뇌의 활성화 부위 해석 및 인사이트 도출을 목표로 한다.
 * Machine Learning / Deep Learning 을 적용한 감정 인식 알고리즘을 개발하고자 한다. 이미 Yu, Minchan [1]에서 SVM 분류기를 통해 73.77%의 정확도로 알고리즘이 개발 된 바 있지만, 해당 논문에서 사용하지 않은 분석 기법을 활용하여 제시된 정확도 보다 높은 알고리즘을 개발하는 것을 목표로 한다.  

# Method
![READMEimage](https://user-images.githubusercontent.com/83329561/171995177-5e927e4f-9ec7-4b7f-ae82-54b0dcfcfd79.png)
(a): 1000Hz EEG신호 채널별 시각화  
(b): Mscoherence를 통한 Network Map 계산 (59 X 59), 이후 6가지 주파수 대역으로 Filtering  
> * Delta: 0.5-4 Hz  
> * Theta: 4-8 Hz  
> * Alpha: 8-13 Hz  
> * Beta: 13-30 Hz  
> * Gamma: 30-49 Hz  
> * High Gamma: 50-80 Hz  

(c): Mst Algorithm을 이용하여 Undirect binary Matrix로 변환  
(d): 그래프 이론 값들을 이용하여 Functional Connectivity Network Feature 계산  
> * Global Measure : Small-Worldness, Characteristic Path Length, Global Efficiency  
> * Local Measure : Clustering Coefficient, Local Efficiency, Eigenvector Centrality, Betweenness Centrality, Node Degree  

(e): ANOVA Test, Gameshowell 통계분석을 통한 유의미한 (* : P<0.05, ** : P<0.01, *** : P<0.001) 채널 도출 및 T-value Head Topologymap Visualization

# Main Code


### Calculate mscoherence Matrix  
* Scipy의 Signal 라이브러리 활용
* Output의 Frequency Sampling값 순서를 인덱스로 저장하여 Filtering 진행 

```python
from scipy import signal
import time
from datetime import datetime

current_time = datetime.now()
print(current_time)

start = time.time()

'''
delta = [0, 2] #index값 0,1.95,3.9 
theta = [3, 4] #5.85, 7.81
alpha = [5, 6] #9.76,11.71
beta = [7, 15] #13.67, 29.29
gamma = [16, 25] #31.25 48.82
high gamma = [52, 83] # 50.78125 80.078125

'''
bandfreq = [[0,5,'delta'],[5,9,'theta'],[9,15,'alpha'],[15,32,'beta'],[32,52,'gamma'],[52,83,'highgamma']]
Valence = [['positive',708],['negative',696],['neutral',639]]


for val in Valence: #positive, negative, neutral
    for band in bandfreq: #delta, theta, alpha, beta, gamma
        globals()['{}_{}_coh_matrixs'.format(val[0], band[2])]=[]
        for i in range(val[1]):
            globals()['{}_{}_cohmat'.format(val[0],band[2])] = [] # 59*59
            for c in range(chan_len):
                globals()['{}_coh'.format(val[0])] =[] # 1*59
                for j in range(chan_len):
                    _, coh = signal.coherence(globals()['{}_filter_EEG'.format(val[0])][i][c],globals()['{}_filter_EEG'.format(val[0])][i][j],fs =1000,window=window)
                    cohb = np.mean(coh[band[0]:band[1]])
                    globals()['{}_coh'.format(val[0])].append(cohb)
                globals()['{}_{}_cohmat'.format(val[0],band[2])].append(globals()['{}_coh'.format(val[0])])
            globals()['{}_{}_coh_matrixs'.format(val[0], band[2])].append(globals()['{}_{}_cohmat'.format(val[0],band[2])])
        print('{}_{}_completed!'.format(val[0], band[2]))

        
        
print("time :", ((time.time() - start)/60)/60 ,"hours")
current_time = datetime.now()
print(current_time)
```
### Convert to Undirect binary matrix using MST algorithm  
* Brainconn 라이브러리 활용
* Mst Algorithm을 활용하여 Undirect Binary Matrix로 변환 (bc.utils.visualization.backbone_wu ==> Mst algorithm 포함한 함수) 

```python

import brainconn as bc
'''
positive_delta_EEG
example)

we use cij 
cij , _ = bc.utils.visualization.backbone_wu(positive_delta_EEG[13],3)

'''
current_time = datetime.now()
print(current_time)
start = time.time()

for fr in bandfreq:
    for num in range(globals()['positive_{}_EEG'.format(fr[2])].shape[0]): # Positive Length:
        _, globals()['Positive_{}_{}_corrcoef_Matrix_binarize'.format(fr[2], num)] = bc.utils.visualization.backbone_wu(globals()['positive_{}_EEG'.format(fr[2])][num], 5)

    for num in range(globals()['negative_{}_EEG'.format(fr[2])].shape[0]): # Negative Length:
        _, globals()['Negative_{}_{}_corrcoef_Matrix_binarize'.format(fr[2],num)]= bc.utils.visualization.backbone_wu(globals()['negative_{}_EEG'.format(fr[2])][num], 5)
        
    for num in range(globals()['neutral_{}_EEG'.format(fr[2])].shape[0]): # Neutral Length:
        _, globals()['Neutral_{}_{}_corrcoef_Matrix_binarize'.format(fr[2],num)]= bc.utils.visualization.backbone_wu(globals()['neutral_{}_EEG'.format(fr[2])][num], 5)
        

print("time :", ((time.time() - start)/60)/60 ,"hours")
current_time = datetime.now()
print(current_time)

```

### Calculate Small-Worldness
* Brainconn 라이브러리 활용
* Small worldness is calculated below formula.
$$  SW = (C/Cr) / (L/Lr) $$ 

1. C is clustering coefficient, L is characteristic path length.  
2. Cr is clustering coefficient of randomized Network, caculated by mean of 100 times iteration. 
3. Lr is characteristic path length of radomized Network, calculated by mean of 100 times iteration.  
> https://link.springer.com/article/10.1007/s00429-015-1035-6  
> https://www.sciencedirect.com/science/article/pii/S1053811916304359  

```python
Cr = np.mean(C_C_random)
Lr = np.mean(C_P_random)

vale = ['Positive','Neutral', 'Negative']
band = ['delta','theta','alpha', 'beta','gamma','highgamma']
leng = [708, 639, 696]


for i, val in enumerate(vale):
    for bd in band:
        globals()['S_W_{}_{}'.format(val, bd)] = []


for i, val in enumerate(vale):
    for bd in band:
        for q in range(leng[i]):
            C = globals()['C_C_{}_{}'.format(val, bd)][q]/Cr
            L = globals()['C_P_{}_{}'.format(val, bd)][q]/Lr
            globals()['S_W_{}_{}'.format(val, bd)].append(C/L)
```

### ANOVA test, Gameshowell 사후테스트
* pingouin 라이브러리 활용
* Welch's Anova 방법 사용

```python
'''
통계분석
Clustering Coefficient

'''
import pingouin as pg

band = ['delta','theta','alpha', 'beta','gamma','highgamma']
cc_anova = {'delta':{},'theta':{},'alpha':{},'beta':{},'gamma':{},'highgamma':{}} #Negative - Neutral {band: {ch : p-value}}
cc_Ttest = {'delta':{},'theta':{},'alpha':{},'beta':{},'gamma':{},'highgamma':{}}

for b in band:
    for ch in ch_names:
        #ANOVA 적용
        a = pg.welch_anova(data = C_C.loc[(C_C['band']==b) & (C_C['ch']==ch)], dv = 'value' ,between ='valence')
        if a['p-unc'][0] < 0.05:
            pw = pg.pairwise_gameshowell(data = C_C.loc[(C_C['band']==b) & (C_C['ch']==ch)], dv = 'value' ,between ='valence')
            pwtest = {'A': pw['pval'][0],'B': pw['pval'][1],'C': pw['pval'][2]}
            cc_anova[b][ch] = pwtest
            ccT = {}
            
            if pw['pval'][0] < 0.05:
                ccT['A'] = pw['T'][0]
            else:
                ccT['A'] = 0
            
            if pw['pval'][1] < 0.05:
                ccT['B'] = pw['T'][1]
            else:
                ccT['B'] = 0
            
            if pw['pval'][2] < 0.05:
                ccT['C'] = pw['T'][2]
            else:
                ccT['C'] = 0
                
            cc_Ttest[b][ch] = ccT
            
        elif a['p-unc'][0] > 0.05:
            ttest = {'A':0,'B':0,'C':0}
            cc_Ttest[b][ch] = ttest
    
```

# Result
* Theta 밴드의 (Occipital) 지역에서 부정적 감정이 중립적, 긍정적 감정보다 낮았음 ==> Poz 채널
* Graph Measure(C.C , B.C …) 방법 중 Node Degree에서 유의한 노드가 가장 많이 검출되었음 ==> 분류 성능이 높을 것으로 예상
* Gamma 밴드의 (Central) 지역에서 부정적 감정이 중립적, 긍정적 감정보다 높았음 ==> CP5 채널
* 전반적으로 부정과 관련된 비교(Neg-Neu, Neg-Pos)에서 유의한 노드가 많이 검출 
* 물론 Neg, Pos와 비교하여 높고 낮음에는 지역적인 부분마다 다르지만, Pos-Neu에 비해 유의한 노드가 많이 나옴

### 3-valence (Positive, Neutral, Negative) comparison에서 모두 유의한 결과가 도출된 채널
* Betweenness Centrality Theta Band PO4 Channel
![READMEimge2](https://user-images.githubusercontent.com/83329561/171998762-fdd28695-0d62-4954-a422-544545b64fb2.png)
* Eigenvector Centraltity Gamma Band O1 Channel:
![READMEimage3](https://user-images.githubusercontent.com/83329561/171999054-099de194-15d2-4b8f-812d-576298b112c9.png)

# Data Acquisition
이 연구는 [1]의 저자에게 요청 후 데이터를 획득받아 수행되었습니다.

# Reference
[1]: Yu, Minchang, et al. "EEG-based emotion recognition in an immersive virtual reality environment: From local activity to brain network features."Biomedical Signal Processing and Control72 (2022): 103349.

