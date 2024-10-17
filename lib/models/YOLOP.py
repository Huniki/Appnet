import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
sys.path.append(os.getcwd())
sys.path.append("lib/models")
sys.path.append("lib/utils")
#sys.path.append("/workspace/wh/projects/DaChuang")
from lib.utils import initialize_weights
# from lib.models.common2 import DepthSeperabelConv2d as Conv
# from lib.models.common2 import SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect
from lib.models.common2 import down_sample,DAttention,TripletAttention,iRMB
from lib.models.common import C3, DetectAux,C2f_EMSC,C2f_EMSCX,C2f_EMSCP,SPPF, SP,SPPELAN,SPPFCSPC, BasicRFB,BasicConv,C2f,Conv, SPP, Bottleneck,BottleneckCSP, Focus, Concat, Detect, SharpenConv, SimSPPF, ELAN,ELAN_X, TB, SimAM, GridMask
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from lib.models.dyhead import *
# sdyx
"""
MCnet_SPP = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ -1, Conv,[512, 256, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1, 6], Concat, [1]],
[ -1, BottleneckCSP, [512, 256, 1, False]],
[ -1, Conv, [256, 128, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,4], Concat, [1]],
[ -1, BottleneckCSP, [256, 128, 1, False]],
[ -1, Conv, [128, 128, 3, 2]],
[ [-1, 14], Concat, [1]],
[ -1, BottleneckCSP, [256, 256, 1, False]],
[ -1, Conv, [256, 256, 3, 2]],
[ [-1, 10], Concat, [1]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
# [ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],
[ [17, 20, 23], Detect,  [13, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],
[ 17, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, SPP, [8, 2, [5, 9, 13]]] #segmentation output
]
# [2,6,3,9,5,13], [7,19,11,26,17,39], [28,64,44,103,61,183]

MCnet_0 = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ -1, Conv,[512, 256, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1, 6], Concat, [1]],
[ -1, BottleneckCSP, [512, 256, 1, False]],
[ -1, Conv, [256, 128, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,4], Concat, [1]],
[ -1, BottleneckCSP, [256, 128, 1, False]],
[ -1, Conv, [128, 128, 3, 2]],
[ [-1, 14], Concat, [1]],
[ -1, BottleneckCSP, [256, 256, 1, False]],
[ -1, Conv, [256, 256, 3, 2]],
[ [-1, 10], Concat, [1]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [8, 2, 3, 1]], #Driving area segmentation output

[ 16, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [8, 2, 3, 1]], #Lane line segmentation output
]


# The lane line and the driving area segment branches share information with each other
MCnet_share = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 64, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ [-1,2], Concat, [1]],  #27
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck

[ 16, Conv, [256, 64, 3, 1]],   #33
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ [-1,2], Concat, [1]], #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39   
[ -1, BottleneckCSP, [16, 8, 1, False]],    #40 lane line segment neck

[ [31,39], Concat, [1]],    #41
[ -1, Conv, [32, 8, 3, 1]],     #42    Share_Block


[ [32,42], Concat, [1]],     #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [16, 2, 3, 1]], #45 Driving area segmentation output


[ [40,42], Concat, [1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [16, 2, 3, 1]] #48Lane line segmentation output
]

# The lane line and the driving area segment branches without share information with each other
MCnet_no_share = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 64, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ [-1,2], Concat, [1]],  #27
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #33
[ -1, Conv, [8, 3, 3, 1]], #34 Driving area segmentation output

[ 16, Conv, [256, 64, 3, 1]],   #35
[ -1, Upsample, [None, 2, 'nearest']],  #36
[ [-1,2], Concat, [1]], #37
[ -1, BottleneckCSP, [128, 64, 1, False]],  #38
[ -1, Conv, [64, 32, 3, 1]],    #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, Conv, [32, 16, 3, 1]],    #41
[ -1, BottleneckCSP, [16, 8, 1, False]],    #42 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [8, 2, 3, 1]] #44 Lane line segmentation output
]

MCnet_feedback = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #33
[ -1, Conv, [8, 2, 3, 1]], #34 Driving area segmentation output

[ 16, Conv, [256, 128, 3, 1]],   #35
[ -1, Upsample, [None, 2, 'nearest']],  #36
[ -1, BottleneckCSP, [128, 64, 1, False]],  #38
[ -1, Conv, [64, 32, 3, 1]],    #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, Conv, [32, 16, 3, 1]],    #41
[ -1, BottleneckCSP, [16, 8, 1, False]],    #42 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [8, 2, 3, 1]] #44 Lane line segmentation output
]


MCnet_Da_feedback1 = [
[46, 26, 35],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16     backbone+fpn
[ -1,Conv,[256,256,1,1]],   #17


[ 16, Conv, [256, 128, 3, 1]],   #18
[ -1, Upsample, [None, 2, 'nearest']],  #19
[ -1, BottleneckCSP, [128, 64, 1, False]],  #20
[ -1, Conv, [64, 32, 3, 1]],    #21
[ -1, Upsample, [None, 2, 'nearest']],  #22
[ -1, Conv, [32, 16, 3, 1]],    #23
[ -1, BottleneckCSP, [16, 8, 1, False]],    #24 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #25
[ -1, Conv, [8, 2, 3, 1]], #26 Driving area segmentation output


[ 16, Conv, [256, 128, 3, 1]],   #27
[ -1, Upsample, [None, 2, 'nearest']],  #28
[ -1, BottleneckCSP, [128, 64, 1, False]],  #29
[ -1, Conv, [64, 32, 3, 1]],    #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, Conv, [32, 16, 3, 1]],    #32
[ -1, BottleneckCSP, [16, 8, 1, False]],    #33 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ -1, Conv, [8, 2, 3, 1]], #35Lane line segmentation output


[ 23, Conv, [16, 16, 3, 2]],     #36
[ -1, Conv, [16, 32, 3, 2]],    #2 times 2xdownsample    37

[ [-1,17], Concat, [1]],       #38
[ -1, BottleneckCSP, [288, 128, 1, False]],    #39
[ -1, Conv, [128, 128, 3, 2]],      #40
[ [-1, 14], Concat, [1]],       #41
[ -1, BottleneckCSP, [256, 256, 1, False]],     #42
[ -1, Conv, [256, 256, 3, 2]],      #43
[ [-1, 10], Concat, [1]],   #44
[ -1, BottleneckCSP, [512, 512, 1, False]],     #45
[ [39, 42, 45], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]] #Detect output 46
]



# The lane line and the driving area segment branches share information with each other and feedback to det_head
MCnet_Da_feedback2 = [
[47, 26, 35],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[25, 28, 31, 33],   #layer in Da_branch to do SAD
[34, 37, 40, 42],   #layer in LL_branch to do SAD
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16     backbone+fpn
[ -1,Conv,[256,256,1,1]],   #17


[ 16, Conv, [256, 128, 3, 1]],   #18
[ -1, Upsample, [None, 2, 'nearest']],  #19
[ -1, BottleneckCSP, [128, 64, 1, False]],  #20
[ -1, Conv, [64, 32, 3, 1]],    #21
[ -1, Upsample, [None, 2, 'nearest']],  #22
[ -1, Conv, [32, 16, 3, 1]],    #23
[ -1, BottleneckCSP, [16, 8, 1, False]],    #24 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #25
[ -1, Conv, [8, 2, 3, 1]], #26 Driving area segmentation output


[ 16, Conv, [256, 128, 3, 1]],   #27
[ -1, Upsample, [None, 2, 'nearest']],  #28
[ -1, BottleneckCSP, [128, 64, 1, False]],  #29
[ -1, Conv, [64, 32, 3, 1]],    #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, Conv, [32, 16, 3, 1]],    #32
[ -1, BottleneckCSP, [16, 8, 1, False]],    #33 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ -1, Conv, [8, 2, 3, 1]], #35Lane line segmentation output


[ 23, Conv, [16, 64, 3, 2]],     #36
[ -1, Conv, [64, 256, 3, 2]],    #2 times 2xdownsample    37

[ [-1,17], Concat, [1]],       #38

[-1, Conv, [512, 256, 3, 1]],     #39
[ -1, BottleneckCSP, [256, 128, 1, False]],    #40
[ -1, Conv, [128, 128, 3, 2]],      #41
[ [-1, 14], Concat, [1]],       #42
[ -1, BottleneckCSP, [256, 256, 1, False]],     #43
[ -1, Conv, [256, 256, 3, 2]],      #44
[ [-1, 10], Concat, [1]],   #45
[ -1, BottleneckCSP, [512, 512, 1, False]],     #46
[ [40, 42, 45], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]] #Detect output 47
]

MCnet_share1 = [
[24, 33, 45],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[25, 28, 31, 33],   #layer in Da_branch to do SAD
[34, 37, 40, 42],   #layer in LL_branch to do SAD
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #27
[ -1, Conv, [64, 32, 3, 1]],    #28
[ -1, Upsample, [None, 2, 'nearest']],  #29
[ -1, Conv, [32, 16, 3, 1]],    #30

[ -1, BottleneckCSP, [16, 8, 1, False]],    #31 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #32
[ -1, Conv, [8, 2, 3, 1]], #33 Driving area segmentation output

[ 16, Conv, [256, 128, 3, 1]],   #34
[ -1, Upsample, [None, 2, 'nearest']],  #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39

[ 30, SharpenConv, [16,16, 3, 1]], #40
[ -1, Conv, [16, 16, 3, 1]], #41
[ [-1, 39], Concat, [1]],   #42
[ -1, BottleneckCSP, [32, 8, 1, False]],    #43 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [8, 2, 3, 1]] #45 Lane line segmentation output
]"""

# 车道线和驾驶区域段在没有相互共享信息的情况下分支，也没有链接
# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
[24, 33, 42],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16         #Encoder

[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24
# 训练多种类别时，需要替换这里的 [17, 20, 23], Detect,  [nc=10/3, [[3,9,5,11,4,20] …… #20230904
[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #27
[ -1, Conv, [64, 32, 3, 1]],    #28
[ -1, Upsample, [None, 2, 'nearest']],  #29
[ -1, Conv, [32, 16, 3, 1]],    #30
[ -1, BottleneckCSP, [16, 8, 1, False]],    #31
[ -1, Upsample, [None, 2, 'nearest']],  #32
[ -1, Conv, [8, 2, 3, 1]], #33 Driving area segmentation head

[ 16, Conv, [256, 128, 3, 1]],   #34
[ -1, Upsample, [None, 2, 'nearest']],  #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39
[ -1, BottleneckCSP, [16, 8, 1, False]],    #40
[ -1, Upsample, [None, 2, 'nearest']],  #41
[ -1, Conv, [8, 2, 3, 1]] #42 Lane line segmentation head
]

YOLOPE = [
[28, 43, 52],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 32, 3, 1]],   #0 
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, Conv, [64, 64, 3, 1]],    #2
[ -1, Conv, [64, 128, 3, 2]],    #3
[ -1, ELAN, [128, 64, 256, 4, 1, [-1, -3, -5, -6]]],  #4
[ -1, TB, [256, 128]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1, ELAN, [256, 128, 512, 4, 1, [-1, -3, -5, -6]]],  #6
[ -1, TB, [512, 256]], # 7
[ -1, ELAN, [512, 256, 1024, 4, 1, [-1, -3, -5, -6]]],  #8
[ -1, TB, [1024, 512]], # 9
[ -1, ELAN, [1024, 256, 1024, 4, 1, [-1, -3, -5, -6]]],  #10
[ -1, SimSPPF, [1024, 512, 5]], #11
# [ -1, SPPFCSPC, [1024, 512]], #11
[ -1, Conv, [512, 256]],   #12
[ -1, Upsample, [None, 2, 'nearest']],  #13
[ 8, Conv, [1024, 256]],   #14
[ [-1, 13], Concat, [1]],    #15
[ -1, ELAN, [512, 128, 256, 4, 2, [-1, -2, -3, -4, -5, -6]]],  #16
[ -1, Conv, [256, 128]],   #17
[ -1, Upsample, [None, 2, 'nearest']],  #18
[ 6, Conv, [512, 128]],   #19
[ [-1, 18], Concat, [1]],    #20
[ -1, ELAN, [256, 64, 128, 4, 2, [-1, -2, -3, -4, -5, -6]]],  #21
[ -1, TB, [128, 128]], # 22
[ [-1, 16], Concat, [1]],    #23
[ -1, ELAN, [512, 128, 256, 4, 2, [-1, -2, -3, -4, -5, -6]]],  #24
[ -1, TB, [256, 256]], # 25
[ [-1, 11], Concat, [1]],    #26
[ -1, ELAN, [1024, 256, 512, 4, 2, [-1, -2, -3, -4, -5, -6]]],  #27
[ [21, 24, 27], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #28 #Detection head 


[ 10, ELAN, [1024, 256, 512, 4, 2, [-1, -2, -3, -4, -5, -6]]],  # 29
[ -1, Conv, [512, 256, 3, 1]],   # 30
[ -1, Upsample, [None, 2, 'nearest']],  # 31
[ -1, ELAN, [256, 128, 256, 4, 2, [-1, -2, -3, -4, -5, -6]]],  # 32
[ -1, Conv, [256, 128, 3, 1]],   # 33
[ -1, Upsample, [None, 2, 'nearest']],  # 34
[ -1, Conv, [128, 128, 3, 1]],   # 35
[ -1, Upsample, [None, 2, 'nearest']],  # 36
[ -1, ELAN, [128, 32, 64, 4, 2, [-1, -2, -3, -4, -5, -6]]],  # 37
[ -1, Conv, [64, 32, 3, 1]],    # 38
[ -1, Upsample, [None, 2, 'nearest']],  # 39
[ -1, Conv, [32, 16, 3, 1]],    # 40
[ -1, ELAN, [16, 4, 8, 4, 2, [-1, -2, -3, -4, -5, -6]]],  # 41
[ -1, Upsample, [None, 2, 'nearest']],  # 42
[ -1, Conv, [8, 2, 3, 1]], # 43 Driving area segmentation head

# [ 10, C3, [1024, 512, 1, False]], # 29
# [ -1, Conv, [512, 256, 3, 1]],   # 30
# [ -1, Upsample, [None, 2, 'nearest']],  # 31
# [ -1, C3, [256, 256, 1, False]], # 32
# [ -1, Conv, [256, 128, 3, 1]],   # 33
# [ -1, Upsample, [None, 2, 'nearest']],  # 34
# [ -1, Conv, [128, 128, 3, 1]],   # 35
# [ -1, Upsample, [None, 2, 'nearest']],  # 36
# [ -1, C3, [128, 64, 1, False]],  # 37
# [ -1, Conv, [64, 32, 3, 1]],    # 38
# [ -1, Upsample, [None, 2, 'nearest']],  # 39
# [ -1, Conv, [32, 16, 3, 1]],    # 40
# [ -1, C3, [16, 8, 1, False]],    # 41
# [ -1, Upsample, [None, 2, 'nearest']],  # 42
# [ -1, Conv, [8, 2, 3, 1]], # 43 Driving area segmentation head


[ 20, Conv, [256, 128, 3, 1]],   #44
[ -1, Upsample, [None, 2, 'nearest']],  #45
[ -1, ELAN, [128, 32, 64, 4, 2, [-1, -2, -3, -4, -5, -6]]],  # 46
[ -1, Conv, [64, 32, 3, 1]],    #47
[ -1, Upsample, [None, 2, 'nearest']],  #48
[ -1, Conv, [32, 16, 3, 1]],    #49
[ -1, ELAN, [16, 4, 8, 4, 2, [-1, -2, -3, -4, -5, -6]]],  # 50
[ -1, Upsample, [None, 2, 'nearest']],  #51
[ -1, Conv, [8, 2, 3, 1]] #52 Lane line segmentation head

# [ 20, Conv, [256, 128, 3, 1]],   #44
# [ -1, Upsample, [None, 2, 'nearest']],  #45
# [ -1, C3, [128, 64, 1, False]],  #46
# [ -1, Conv, [64, 32, 3, 1]],    #47
# [ -1, Upsample, [None, 2, 'nearest']],  #48
# [ -1, Conv, [32, 16, 3, 1]],    #49
# [ -1, C3, [16, 8, 1, False]],    #50
# [ -1, Upsample, [None, 2, 'nearest']],  #51
# [ -1, Conv, [8, 2, 3, 1]] #52 Lane line segmentation head
]
YOLOPV8 = [
[23, 38, 47],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f, [512, 512,1,True]],  #8
# [ -1, SimSPPF, [512, 512, 5]], #9
[ -1, SPPFCSPC, [512, 512]], #9


[ -1, Upsample, [None, 2, 'nearest']],  #10  
[ [-1, 6], Concat, [1]],    #11    1024
[ -1,  C2f, [1024, 512,1,True]],#12       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#13  80 80 512
[ -1,  C2f, [512, 256,1,True]],#14   80 80 256
[ [-1, 4], Concat, [1]],    #15  80  80 512#encoder

[ -1,  C2f, [512,512,1,True]],#116  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #17
[ [-1, 12], Concat, [1]],   #18  1024
[ -1,  C2f, [1024,512,1,True]],#19 YOLOHEAD2
[ -1, down_sample, [512,512]],  #20
[ [-1, 9], Concat, [1]],   #21
[ -1,  C2f, [1024,1024,1,True]],#22  YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 7, C2f, [512, 512,1,True]],  # 24
[ -1, Conv, [512, 256, 3, 1]],   # 25
[ -1, Upsample, [None, 2, 'nearest']],  # 26
[ -1, C2f, [256, 256,1,True]],  # 27
[ -1, Conv, [256, 128, 3, 1]],   # 28
[ -1, Upsample, [None, 2, 'nearest']],  # 29
[ -1, Conv, [128, 128, 3, 1]],   # 30
[ -1, Upsample, [None, 2, 'nearest']],  # 31
[ -1, C2f, [128,64,1,True]],  # 32
[ -1, Conv, [64, 32, 3, 1]],    # 33
[ -1, Upsample, [None, 2, 'nearest']],  # 34
[ -1, Conv, [32, 16, 3, 1]],    # 35
[ -1, C2f, [16, 8,1,True]],  # 36
[ -1, Upsample, [None, 2, 'nearest']],  # 37
[ -1, Conv, [8, 2, 3, 1]], # 38 Driving area segmentation head


[ 13, Conv, [512, 128, 3, 1]],   #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, C2f, [128, 64]],  # 41
[ -1, Conv, [64, 32, 3, 1]],    #42
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [32, 16, 3, 1]],    #44
[ -1, C2f, [16,  8]],  # 45
[ -1, Upsample, [None, 2, 'nearest']],  #46
[ -1, Conv, [8, 2, 3, 1]] #47 Lane line segmentation head
]
YOLOPV8_attention_cat_01 = [
[27, 42, 51],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f_EMSC, [512, 512,3,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f_EMSC, [512, 512,3,True]],  #8
[ -1, SimSPPF, [512, 512, 5]], #9
[ -1, SimAM, [1, 1e-4]],#10
# [-1, DAttention,[[20,20]]],
# [-1,iRMB,[512,512]],
# [-1,TripletAttention,[]],

[ -1, Upsample, [None, 2, 'nearest']],  #11 
[ [-1, 6], Concat, [1]],    #12    1024
[ -1,  C2f, [1024, 512,1,True]],#13       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#14  80 80 512
[ -1,  C2f, [512, 256,1,True]],#15   80 80 256
[ [-1, 4], Concat, [1]],    #16  80  80 512#encoder


[ -1,  C2f_EMSC, [512,512,3,True]],#17
[ -1, SimAM, [1, 1e-4]],                    #18YOLOHEAD1   

[ -1, down_sample, [512,512]],  #19
[ [-1, 13], Concat, [1]],#20
[ -1,  C2f_EMSC, [1024,512,3,True]], #21
[ -1, SimAM, [1, 1e-4]],#22       YOLOHEAD2
 
[ -1, down_sample, [512,512]],  #23
[ [-1, 10], Concat, [1]],   #24
[ -1,  C2f_EMSC, [1024,1024,3,True]],#25
[ -1, SimAM, [1, 1e-4]], #  #26                YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [18, 20, 26], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 9, C2f, [512, 512,1,True]],  # 28
[ -1, Conv, [512, 256, 3, 1]],   # 29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, C2f, [256, 256,1,True]],  # 31
[ -1, Conv, [256, 128, 3, 1]],   # 32
[ -1, Upsample, [None, 2, 'nearest']],  # 33
[ -1, Conv, [128, 128, 3, 1]],   # 34
[ -1, Upsample, [None, 2, 'nearest']],  # 35
[ -1, C2f, [128,64,1,True]],  # 36
[ -1, Conv, [64, 32, 3, 1]],    # 37
[ -1, Upsample, [None, 2, 'nearest']],  # 38
[ -1, Conv, [32, 16, 3, 1]],    # 39
[ -1, C2f, [16, 8,1,True]],  # 40
[ -1, Upsample, [None, 2, 'nearest']],  # 41
[ -1, Conv, [8, 2, 3, 1]], # 42 Driving area segmentation head


[ 14, Conv, [512, 128, 3, 1]],   #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, C2f, [128, 64]],  # 45
[ -1, Conv, [64, 32, 3, 1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [32, 16, 3, 1]],    #48
[ -1, C2f, [16,  8]],  # 49
[ -1, Upsample, [None, 2, 'nearest']],  #50
[ -1, Conv, [8, 2, 3, 1]] #51 Lane line segmentation head
]

YOLOPV8_attention_cat2 = [
[27, 42, 51],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f_EMSCX, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f_EMSCX, [512, 512,1,True]],  #8
[ -1, SimSPPF, [512, 512, 5]], #9
[ -1, SimAM, [1, 1e-4]],#10
# [-1, DAttention,[[20,20]]],
# [-1,iRMB,[512,512]],
# [-1,TripletAttention,[]],

[ -1, Upsample, [None, 2, 'nearest']],  #11 
[ [-1, 6], Concat, [1]],    #12    1024
[ -1,  C2f, [1024, 512,1,True]],#13       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#14  80 80 512
[ -1,  C2f, [512, 256,1,True]],#15   80 80 256
[ [-1, 4], Concat, [1]],    #16  80  80 512#encoder


[ -1,  C2f_EMSCX, [512,512,1,True]],#17
[ -1, SimAM, [1, 1e-4]],                    #18YOLOHEAD1   

[ -1, down_sample, [512,512]],  #19
[ [-1, 13], Concat, [1]],#20
[ -1,  C2f_EMSCX, [1024,1024,1,True]], #21
[ -1, SimAM, [1, 1e-4]],#22       YOLOHEAD2
 
[ -1, down_sample, [1024,512]],  #23
[ [-1, 9], Concat, [1]],   #24
[ -1,  C2f_EMSCX, [1024,1024,1,True]],#25
[ -1, SimAM, [1, 1e-4]], #  #26                YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [18, 22, 26], Detect,  [1, [[1,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 9, C2f, [512, 512,1,True]],  # 28
[ -1, Conv, [512, 256, 3, 1]],   # 29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, C2f, [256, 256,1,True]],  # 31
[ -1, Conv, [256, 128, 3, 1]],   # 32
[ -1, Upsample, [None, 2, 'nearest']],  # 33
[ -1, Conv, [128, 128, 3, 1]],   # 34
[ -1, Upsample, [None, 2, 'nearest']],  # 35
[ -1, C2f, [128,64,1,True]],  # 36
[ -1, Conv, [64, 32, 3, 1]],    # 37
[ -1, Upsample, [None, 2, 'nearest']],  # 38
[ -1, Conv, [32, 16, 3, 1]],    # 39
[ -1, C2f, [16, 8,1,True]],  # 40
[ -1, Upsample, [None, 2, 'nearest']],  # 41
[ -1, Conv, [8, 2, 3, 1]], # 42 Driving area segmentation head


[ 14, Conv, [512, 128, 3, 1]],   #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, C2f, [128, 64]],  # 45
[ -1, Conv, [64, 32, 3, 1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [32, 16, 3, 1]],    #48
[ -1, C2f, [16,  8]],  # 49
[ -1, Upsample, [None, 2, 'nearest']],  #50
[ -1, Conv, [8, 2, 3, 1]] #51 Lane line segmentation head
]

YOLOPV8_sppfChange= [
[23, 38, 47],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f, [512, 512,1,True]],  #8
# [ -1, SimSPPF, [512, 512, 5]], #9
# [ -1, SPPFCSPC, [512, 512]], #9
# [ -1, BasicRFB, [512, 512]],
[ -1, SPPELAN, [512, 512,1024]],#c3参数的选取还需待定


[ -1, Upsample, [None, 2, 'nearest']],  #10  
[ [-1, 6], Concat, [1]],    #11    1024
[ -1,  C2f, [1024, 512,1,True]],#12       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#13  80 80 512
[ -1,  C2f, [512, 256,1,True]],#14   80 80 256
[ [-1, 4], Concat, [1]],    #15  80  80 512#encoder

[ -1,  C2f, [512,512,1,True]],#116  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #17
[ [-1, 12], Concat, [1]],   #18  1024
[ -1,  C2f, [1024,512,1,True]],#19 YOLOHEAD2
[ -1, down_sample, [512,512]],  #20
[ [-1, 9], Concat, [1]],   #21
[ -1,  C2f, [1024,1024,1,True]],#22  YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 7, C2f, [512, 512,1,True]],  # 24
[ -1, Conv, [512, 256, 3, 1]],   # 25
[ -1, Upsample, [None, 2, 'nearest']],  # 26
[ -1, C2f, [256, 256,1,True]],  # 27
[ -1, Conv, [256, 128, 3, 1]],   # 28
[ -1, Upsample, [None, 2, 'nearest']],  # 29
[ -1, Conv, [128, 128, 3, 1]],   # 30
[ -1, Upsample, [None, 2, 'nearest']],  # 31
[ -1, C2f, [128,64,1,True]],  # 32
[ -1, Conv, [64, 32, 3, 1]],    # 33
[ -1, Upsample, [None, 2, 'nearest']],  # 34
[ -1, Conv, [32, 16, 3, 1]],    # 35
[ -1, C2f, [16, 8,1,True]],  # 36
[ -1, Upsample, [None, 2, 'nearest']],  # 37
[ -1, Conv, [8, 2, 3, 1]], # 38 Driving area segmentation head


[ 13, Conv, [512, 128, 3, 1]],   #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, C2f, [128, 64]],  # 41
[ -1, Conv, [64, 32, 3, 1]],    #42
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [32, 16, 3, 1]],    #44
[ -1, C2f, [16,  8]],  # 45
[ -1, Upsample, [None, 2, 'nearest']],  #46
[ -1, Conv, [8, 2, 3, 1]] #47 Lane line segmentation head
]

YOLOPV8_cf2_emscp = [
[23, 38, 47],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f_EMSCP, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f_EMSCP, [512, 512,1,True]],  #8
[ -1, SimSPPF, [512, 512, 5]], #9
# [ -1, SPPFCSPC, [1024, 512]], #9

[ -1, Upsample, [None, 2, 'nearest']],  #10  
[ [-1, 6], Concat, [1]],    #11    1024
[ -1,  C2f, [1024, 512,1,True]],#12       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#13  80 80 512
[ -1,  C2f, [512, 256,1,True]],#14   80 80 256
[ [-1, 4], Concat, [1]],    #15  80  80 512#encoder

[ -1,  C2f_EMSCP, [512,512,1,True]],#116  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #17
[ [-1, 12], Concat, [1]],   #18  1024
[ -1,  C2f_EMSCP, [1024,512,1,True]],#19 YOLOHEAD2
[ -1, down_sample, [512,512]],  #20
[ [-1, 9], Concat, [1]],   #21
[ -1,  C2f_EMSCP, [1024,1024,1,True]],#22  YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 7, C2f, [512, 512,1,True]],  # 24
[ -1, Conv, [512, 256, 3, 1]],   # 25
[ -1, Upsample, [None, 2, 'nearest']],  # 26
[ -1, C2f, [256, 256,1,True]],  # 27
[ -1, Conv, [256, 128, 3, 1]],   # 28
[ -1, Upsample, [None, 2, 'nearest']],  # 29
[ -1, Conv, [128, 128, 3, 1]],   # 30
[ -1, Upsample, [None, 2, 'nearest']],  # 31
[ -1, C2f, [128,64,1,True]],  # 32
[ -1, Conv, [64, 32, 3, 1]],    # 33
[ -1, Upsample, [None, 2, 'nearest']],  # 34
[ -1, Conv, [32, 16, 3, 1]],    # 35
[ -1, C2f, [16, 8,1,True]],  # 36
[ -1, Upsample, [None, 2, 'nearest']],  # 37
[ -1, Conv, [8, 2, 3, 1]], # 38 Driving area segmentation head


[ 13, Conv, [512, 128, 3, 1]],   #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, C2f, [128, 64]],  # 41
[ -1, Conv, [64, 32, 3, 1]],    #42
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [32, 16, 3, 1]],    #44
[ -1, C2f, [16,  8]],  # 45
[ -1, Upsample, [None, 2, 'nearest']],  #46
[ -1, Conv, [8, 2, 3, 1]] #47 Lane line segmentation head
]

YOLOPV8_cf2_emsc = [
[23, 38, 47],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f_EMSC, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f_EMSC, [512, 512,1,True]],  #8
[ -1, SimSPPF, [512, 512, 5]], #9
# [ -1, SPPFCSPC, [1024, 512]], #9

[ -1, Upsample, [None, 2, 'nearest']],  #10  
[ [-1, 6], Concat, [1]],    #11    1024
[ -1,  C2f, [1024, 512,1,True]],#12       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#13  80 80 512
[ -1,  C2f, [512, 256,1,True]],#14   80 80 256
[ [-1, 4], Concat, [1]],    #15  80  80 512#encoder

[ -1,  C2f_EMSC, [512,512,1,True]],#116  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #17
[ [-1, 12], Concat, [1]],   #18  1024
[ -1,  C2f_EMSC, [1024,512,1,True]],#19 YOLOHEAD2
[ -1, down_sample, [512,512]],  #20
[ [-1, 9], Concat, [1]],   #21
[ -1,  C2f_EMSC, [1024,1024,1,True]],#22  YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 

[ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 7, C2f, [512, 512,1,True]],  # 24
[ -1, Conv, [512, 256, 3, 1]],   # 25
[ -1, Upsample, [None, 2, 'nearest']],  # 26
[ -1, C2f, [256, 256,1,True]],  # 27
[ -1, Conv, [256, 128, 3, 1]],   # 28
[ -1, Upsample, [None, 2, 'nearest']],  # 29
[ -1, Conv, [128, 128, 3, 1]],   # 30
[ -1, Upsample, [None, 2, 'nearest']],  # 31
[ -1, C2f, [128,64,1,True]],  # 32
[ -1, Conv, [64, 32, 3, 1]],    # 33
[ -1, Upsample, [None, 2, 'nearest']],  # 34
[ -1, Conv, [32, 16, 3, 1]],    # 35
[ -1, C2f, [16, 8,1,True]],  # 36
[ -1, Upsample, [None, 2, 'nearest']],  # 37
[ -1, Conv, [8, 2, 3, 1]], # 38 Driving area segmentation head


[ 13, Conv, [512, 128, 3, 1]],   #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, C2f, [128, 64]],  # 41
[ -1, Conv, [64, 32, 3, 1]],    #42
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [32, 16, 3, 1]],    #44
[ -1, C2f, [16,  8]],  # 45
[ -1, Upsample, [None, 2, 'nearest']],  #46
[ -1, Conv, [8, 2, 3, 1]] #47 Lane line segmentation head
]

YOLOPV8_attention_DAttention = [
[27, 42, 51],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f, [512, 512,1,True]],  #8
[ -1, SPPFCSPC, [1024, 512]], #9
# [ -1, SimAM, [1, 1e-4]],#10
[-1, DAttention,[[20,20]]],
# [-1,iRMB,[512,512]],
# [-1,TripletAttention,[]],

[ -1, Upsample, [None, 2, 'nearest']],  #11 
[ [-1, 6], Concat, [1]],    #12    1024
[ -1,  C2f, [1024, 512,1,True]],#13       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#14  80 80 512
[ -1,  C2f, [512, 256,1,True]],#15   80 80 256
[ [-1, 4], Concat, [1]],    #16  80  80 512#encoder

[-1, DAttention,[[20,20]]], #17

[ -1,  C2f, [512,512,1,True]],#18  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #19

[-1, DAttention,[[20,20]]], #20
[ [-1, 13], Concat, [1]],   #21  1024
[ -1,  C2f, [1024,512,1,True]],#22 YOLOHEAD2
[ -1, down_sample, [512,512]],  #23
[ [-1, 9], Concat, [1]],   #24
[-1, DAttention,[[20,20]]], #25
[ -1,  C2f, [1024,1024,1,True]],#26  YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [18, 21, 26], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 7, C2f, [512, 512,1,True]],  # 28
[ -1, Conv, [512, 256, 3, 1]],   # 29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, C2f, [256, 256,1,True]],  # 31
[ -1, Conv, [256, 128, 3, 1]],   # 32
[ -1, Upsample, [None, 2, 'nearest']],  # 33
[ -1, Conv, [128, 128, 3, 1]],   # 34
[ -1, Upsample, [None, 2, 'nearest']],  # 35
[ -1, C2f, [128,64,1,True]],  # 36
[ -1, Conv, [64, 32, 3, 1]],    # 37
[ -1, Upsample, [None, 2, 'nearest']],  # 38
[ -1, Conv, [32, 16, 3, 1]],    # 39
[ -1, C2f, [16, 8,1,True]],  # 40
[ -1, Upsample, [None, 2, 'nearest']],  # 41
[ -1, Conv, [8, 2, 3, 1]], # 42 Driving area segmentation head


[ 14, Conv, [512, 128, 3, 1]],   #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, C2f, [128, 64]],  # 45
[ -1, Conv, [64, 32, 3, 1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [32, 16, 3, 1]],    #48
[ -1, C2f, [16,  8]],  # 49
[ -1, Upsample, [None, 2, 'nearest']],  #50
[ -1, Conv, [8, 2, 3, 1]] #51 Lane line segmentation head
]
YOLOPV8_attention_iRMB = [
[27, 42, 51],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f, [512, 512,1,True]],  #8
[ -1, SPPFCSPC, [512, 512]], #9
# [ -1, SimAM, [1, 1e-4]],#10
# [-1, DAttention,[[20,20]]],
[-1,iRMB,[512,512]],
# [-1,TripletAttention,[]],

[ -1, Upsample, [None, 2, 'nearest']],  #11 
[ [-1, 6], Concat, [1]],    #12    1024
[ -1,  C2f, [1024, 512,1,True]],#13       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#14  80 80 512
[ -1,  C2f, [512, 256,1,True]],#15   80 80 256
[ [-1, 4], Concat, [1]],    #16  80  80 512#encoder

[-1,iRMB,[512,512]], #17

[ -1,  C2f, [512,512,1,True]],#18  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #19

[-1,iRMB,[512,512]], #20
[ [-1, 13], Concat, [1]],   #21  1024
[ -1,  C2f, [1024,512,1,True]],#22 YOLOHEAD2
[ -1, down_sample, [512,512]],  #23
[ [-1, 9], Concat, [1]],   #24
[-1,iRMB,[1024,1024]], #25
[ -1,  C2f, [1024,1024,1,True]],#26  YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [18, 21, 26], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 9, C2f, [512, 512,1,True]],  # 28
[ -1, Conv, [512, 256, 3, 1]],   # 29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, C2f, [256, 256,1,True]],  # 31
[ -1, Conv, [256, 128, 3, 1]],   # 32
[ -1, Upsample, [None, 2, 'nearest']],  # 33
[ -1, Conv, [128, 128, 3, 1]],   # 34
[ -1, Upsample, [None, 2, 'nearest']],  # 35
[ -1, C2f, [128,64,1,True]],  # 36
[ -1, Conv, [64, 32, 3, 1]],    # 37
[ -1, Upsample, [None, 2, 'nearest']],  # 38
[ -1, Conv, [32, 16, 3, 1]],    # 39
[ -1, C2f, [16, 8,1,True]],  # 40
[ -1, Upsample, [None, 2, 'nearest']],  # 41
[ -1, Conv, [8, 2, 3, 1]], # 42 Driving area segmentation head


[ 14, Conv, [512, 128, 3, 1]],   #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, C2f, [128, 64]],  # 45
[ -1, Conv, [64, 32, 3, 1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [32, 16, 3, 1]],    #48
[ -1, C2f, [16,  8]],  # 49
[ -1, Upsample, [None, 2, 'nearest']],  #50
[ -1, Conv, [8, 2, 3, 1]] #51 Lane line segmentation head
]
YOLOPV8_attention_SimAM = [
[27, 42, 51],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f, [512, 512,1,True]],  #8
[ -1, SPPFCSPC, [512, 512]], #9
[ -1, SimAM, [1, 1e-4]],#10
# [-1, DAttention,[[20,20]]],
# [-1,iRMB,[512,512]],
# [-1,TripletAttention,[]],

[ -1, Upsample, [None, 2, 'nearest']],  #11 
[ [-1, 6], Concat, [1]],    #12    1024
[ -1,  C2f, [1024, 512,1,True]],#13       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#14  80 80 512
[ -1,  C2f, [512, 256,1,True]],#15   80 80 256
[ [-1, 4], Concat, [1]],    #16  80  80 512#encoder

[ -1, SimAM, [1, 1e-4]], #17

[ -1,  C2f, [512,512,1,True]],#18  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #19

[ -1, SimAM, [1, 1e-4]], #20
[ [-1, 13], Concat, [1]],   #21  1024
[ -1,  C2f, [1024,512,1,True]],#22 YOLOHEAD2
[ -1, down_sample, [512,512]],  #23
[ [-1, 9], Concat, [1]],   #24
[ -1, SimAM, [1, 1e-4]], #25
[ -1,  C2f, [1024,1024,1,True]],#26  YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [18, 21, 26], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 7, C2f, [512, 512,1,True]],  # 28
[ -1, Conv, [512, 256, 3, 1]],   # 29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, C2f, [256, 256,1,True]],  # 31
[ -1, Conv, [256, 128, 3, 1]],   # 32
[ -1, Upsample, [None, 2, 'nearest']],  # 33
[ -1, Conv, [128, 128, 3, 1]],   # 34
[ -1, Upsample, [None, 2, 'nearest']],  # 35
[ -1, C2f, [128,64,1,True]],  # 36
[ -1, Conv, [64, 32, 3, 1]],    # 37
[ -1, Upsample, [None, 2, 'nearest']],  # 38
[ -1, Conv, [32, 16, 3, 1]],    # 39
[ -1, C2f, [16, 8,1,True]],  # 40
[ -1, Upsample, [None, 2, 'nearest']],  # 41
[ -1, Conv, [8, 2, 3, 1]], # 42 Driving area segmentation head


[ 14, Conv, [512, 128, 3, 1]],   #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, C2f, [128, 64]],  # 45
[ -1, Conv, [64, 32, 3, 1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [32, 16, 3, 1]],    #48
[ -1, C2f, [16,  8]],  # 49
[ -1, Upsample, [None, 2, 'nearest']],  #50
[ -1, Conv, [8, 2, 3, 1]] #51 Lane line segmentation head
]
YOLOPV8_attention_hand = [
[24, 39, 48],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f, [512, 512,1,True]],  #8
[ -1, SimSPPF, [512, 512, 5]], #9
# [ -1, SimAM, [1, 1e-4]],#10
[-1, DAttention,[[20,20]]],
# [-1,iRMB,[512,512]],
# [-1,TripletAttention,[]],

[ -1, Upsample, [None, 2, 'nearest']],  #11 
[ [-1, 6], Concat, [1]],    #12    1024
[ -1,  C2f, [1024, 512,1,True]],#13       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#14  80 80 512
[ -1,  C2f, [512, 256,1,True]],#15   80 80 256
[ [-1, 4], Concat, [1]],    #16  80  80 512#encoder

[ -1,  C2f, [512,512,1,True]],#117  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #18
[ [-1, 13], Concat, [1]],   #19  1024
[ -1,  C2f, [1024,512,1,True]],#20 YOLOHEAD2
[ -1, down_sample, [512,512]],  #21
[ [-1, 9], Concat, [1]],   #22
[ -1,  C2f, [1024,1024,1,True]],#23  YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [17, 19, 23], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 7, C2f, [512, 512,1,True]],  # 25
[ -1, Conv, [512, 256, 3, 1]],   # 26
[ -1, Upsample, [None, 2, 'nearest']],  # 27
[ -1, C2f, [256, 256,1,True]],  # 28
[ -1, Conv, [256, 128, 3, 1]],   # 29
[ -1, Upsample, [None, 2, 'nearest']],  # 30
[ -1, Conv, [128, 128, 3, 1]],   # 31
[ -1, Upsample, [None, 2, 'nearest']],  # 32
[ -1, C2f, [128,64,1,True]],  # 33
[ -1, Conv, [64, 32, 3, 1]],    # 34
[ -1, Upsample, [None, 2, 'nearest']],  # 35
[ -1, Conv, [32, 16, 3, 1]],    # 36
[ -1, C2f, [16, 8,1,True]],  # 37
[ -1, Upsample, [None, 2, 'nearest']],  # 38
[ -1, Conv, [8, 2, 3, 1]], # 39 Driving area segmentation head


[ 14, Conv, [512, 128, 3, 1]],   #40
[ -1, Upsample, [None, 2, 'nearest']],  #441
[ -1, C2f, [128, 64]],  # 42
[ -1, Conv, [64, 32, 3, 1]],    #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [32, 16, 3, 1]],    #45
[ -1, C2f, [16,  8]],  # 46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [8, 2, 3, 1]] #48 Lane line segmentation head
]
YOLOPV8_attention_trip= [
[27, 42, 51],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f, [512, 512,1,True]],  #8
[ -1, SPPFCSPC, [512, 512]], #9
# [ -1, SimAM, [1, 1e-4]],#10
# [-1, DAttention,[[20,20]]],
# # [-1,iRMB,[512,512]],
[-1,TripletAttention,[]],

[ -1, Upsample, [None, 2, 'nearest']],  #11 
[ [-1, 6], Concat, [1]],    #12    1024
[ -1,  C2f, [1024, 512,1,True]],#13       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#14  80 80 512
[ -1,  C2f, [512, 256,1,True]],#15   80 80 256
[ [-1, 4], Concat, [1]],    #16  80  80 512#encoder

[-1,TripletAttention,[]],

[ -1,  C2f, [512,512,1,True]],#18  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #19

[-1,TripletAttention,[]], #20
[ [-1, 13], Concat, [1]],   #21  1024
[ -1,  C2f, [1024,512,1,True]],#22 YOLOHEAD2
[ -1, down_sample, [512,512]],  #23
[ [-1, 9], Concat, [1]],   #24
[-1,TripletAttention,[]], #25
[ -1,  C2f, [1024,1024,1,True]],#26  YOLOHEAD3
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
[ [18, 21, 26], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 7, C2f, [512, 512,1,True]],  # 28
[ -1, Conv, [512, 256, 3, 1]],   # 29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, C2f, [256, 256,1,True]],  # 31
[ -1, Conv, [256, 128, 3, 1]],   # 32
[ -1, Upsample, [None, 2, 'nearest']],  # 33
[ -1, Conv, [128, 128, 3, 1]],   # 34
[ -1, Upsample, [None, 2, 'nearest']],  # 35
[ -1, C2f, [128,64,1,True]],  # 36
[ -1, Conv, [64, 32, 3, 1]],    # 37
[ -1, Upsample, [None, 2, 'nearest']],  # 38
[ -1, Conv, [32, 16, 3, 1]],    # 39
[ -1, C2f, [16, 8,1,True]],  # 40
[ -1, Upsample, [None, 2, 'nearest']],  # 41
[ -1, Conv, [8, 2, 3, 1]], # 42 Driving area segmentation head


[ 14, Conv, [512, 128, 3, 1]],   #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, C2f, [128, 64]],  # 45
[ -1, Conv, [64, 32, 3, 1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [32, 16, 3, 1]],    #48
[ -1, C2f, [16,  8]],  # 49
[ -1, Upsample, [None, 2, 'nearest']],  #50
[ -1, Conv, [8, 2, 3, 1]] #51 Lane line segmentation head
]

YOLOPV8_detect = [
[23, 38, 47],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx # 目标检测层、可驾驶区域检测层、车道线检测层
[ -1, Conv, [3, 64, 3, 2]],   #0 
[ -1, Conv, [64, 128, 3, 2]],    #1
[ -1, C2f, [128, 128,1,True]],    #2   cps模块
[ -1, Conv, [128, 256, 3, 2]],    #3
[ -1, C2f, [256, 256,1,True]],  #4
[ -1, Conv, [256, 512, 3, 2]], # 5 # TB的参数是[输入通道数，输出通道数的一半]
[ -1,  C2f, [512, 512,1,True]],  #6
[ -1, Conv, [512, 512, 3, 2]], # 7
[ -1,  C2f, [512, 512,1,True]],  #8
[ -1, SimSPPF, [512, 512, 5]], #9
# [ -1, SPPFCSPC, [1024, 512]], #9

[ -1, Upsample, [None, 2, 'nearest']],  #10  
[ [-1, 6], Concat, [1]],    #11    1024
[ -1,  C2f, [1024, 512,1,True]],#12       40 40 512
[ -1, Upsample, [None, 2, 'nearest']],#13  80 80 512
[ -1,  C2f, [512, 256,1,True]],#14   80 80 256
[ [-1, 4], Concat, [1]],    #15  80  80 512#encoder

[ -1,  C2f, [512,512,1,True]],#116  YOLOHEAD1  
[ -1, down_sample, [512,512]],  #17
[ [-1, 12], Concat, [1]],   #18  1024
[ -1,  C2f, [1024,512,1,True]],#19 YOLOHEAD2
[ -1, down_sample, [512,512]],  #20
[ [-1, 9], Concat, [1]],   #21
[ -1,  C2f, [1024,1024,1,True]],#22  YOLOHEAD3
#[ [16, 18, 22],Detect_DyHead,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #23 #Detection head 
#[ [16, 18, 22],DetectAux,  [10,[[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512,1024,1024]]], #23 #Detection head 
[[16,18,22],Detect_DyHead,[10,128,1, [512,1024,1024]]],
# [ [16, 18, 22], Detect,  [10, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [512, 1024, 1024]]], #23 #Detection head 

####
[ 7, C2f, [512, 512,1,True]],  # 24
[ -1, Conv, [512, 256, 3, 1]],   # 25
[ -1, Upsample, [None, 2, 'nearest']],  # 26
[ -1, C2f, [256, 256,1,True]],  # 27
[ -1, Conv, [256, 128, 3, 1]],   # 28
[ -1, Upsample, [None, 2, 'nearest']],  # 29
[ -1, Conv, [128, 128, 3, 1]],   # 30
[ -1, Upsample, [None, 2, 'nearest']],  # 31
[ -1, C2f, [128,64,1,True]],  # 32
[ -1, Conv, [64, 32, 3, 1]],    # 33
[ -1, Upsample, [None, 2, 'nearest']],  # 34
[ -1, Conv, [32, 16, 3, 1]],    # 35
[ -1, C2f, [16, 8,1,True]],  # 36
[ -1, Upsample, [None, 2, 'nearest']],  # 37
[ -1, Conv, [8, 2, 3, 1]], # 38 Driving area segmentation head


[ 13, Conv, [512, 128, 3, 1]],   #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, C2f, [128, 64]],  # 41
[ -1, Conv, [64, 32, 3, 1]],    #42
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [32, 16, 3, 1]],    #44
[ -1, C2f, [16,  8]],  # 45
[ -1, Upsample, [None, 2, 'nearest']],  #46
[ -1, Conv, [8, 2, 3, 1]] #47 Lane line segmentation head
]



class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1 # 20230904
        self.detector_index = -1
        # 第3次修改，修改可驾驶区域检测头
        self.det_out_idx = block_cfg[0][0] # 24 # 目标检测层
        self.seg_out_idx = block_cfg[0][1:] # [33, 42] # 可驾驶区域检测层、车道线检测层
        # self.det_out_idx = 24
        # self.seg_out_idx = [39,48]
        

        # Build model # 以下注释为第一次循环
        for i, (from_, block, args) in enumerate(block_cfg[1:]): # i=0 # from_ = -1 # block = <class 'lib.models.common.Focus'> # args = [3, 32, 3]
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args) # block_ = Focus(  (conv): Conv(    (conv): Conv2d(12, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)    (act): Hardswish()  ))
            block_.index, block_.from_ = i, from_
            layers.append(block_) # 添加该层 # layers = [Focus(  (conv): Con...sh()  ))]
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist # 附加到保存列表
        assert self.detector_index == block_cfg[0][0] # self.detector_index = 24 # 目标检测层

        self.model, self.save = nn.Sequential(*layers), sorted(save) # self.model = YOLO… # self.save = [4, 6, 10, 14, 16, 16, 17, 20, 23] 需要保存的层
        self.names = [str(i) for i in range(self.nc)] # self.names = ['0']

        # set stride、anchor for detector # stride 步长 # 设置检测器的步长和框
        Detector = self.model[self.detector_index]  # detector 24检测层 # Detector = Detect(  (m): ModuleList(    (0): Conv2d(128, 18, kernel_size=(1, 1), stride=(1, 1))    (1): Conv2d(256, 18, kernel_size=(1, 1), stride=(1, 1))    (2): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))  ))
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s)) # model_out = [[tensor([[[[[ 0.0181,...0105]]]]]), tensor([[[[[-0.0260,...0021]]]]]), tensor([[[[[ 0.0094,...0232]]]]])],      tensor([[[[0.5000, 0....5000]]]]),      tensor([[[[0.5000, 0....5000]]]])]
                detects, _, _= model_out # detects = [tensor([[[[[ 0.0181,...0105]]]]]), tensor([[[[[-0.0260,...0021]]]]]), tensor([[[[[ 0.0094,...0232]]]]])]
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward # Detector.stride = tensor([ 8., 16., 32.])
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale # Detector.anchors = tensor([[[0.3750, 1.1250],         [0.6250, 1.3750],         [0.5000, 2.5000]],        [[0.4375, 1.1250],         [0.3750, 2.4375],         [0.7500, 1.9375]],        [[0.5938, 1.5625],         [1.1875, 2.5312],         [2.1250, 4.9062]]])
            check_anchor_order(Detector)
            self.stride = Detector.stride # self.stride = tensor([ 8., 16., 32.])
            self._initialize_biases()
            

        elif isinstance(Detector, Detect_DyHead):
        #     s = 640  # 2x min stride
        #     with torch.no_grad():
        #         model_out = self.forward(torch.zeros(1, 3, s, s)) # model_out = [[tensor([[[[[ 0.0181,...0105]]]]]), tensor([[[[[-0.0260,...0021]]]]]), tensor([[[[[ 0.0094,...0232]]]]])],      tensor([[[[0.5000, 0....5000]]]]),      tensor([[[[0.5000, 0....5000]]]])]
        #         detects, _, _= model_out # detects = [tensor([[[[[ 0.0181,...0105]]]]]), tensor([[[[[-0.0260,...0021]]]]]), tensor([[[[[ 0.0094,...0232]]]]])]
        #         x =  self.forward(torch.zeros(2,3,s,s))
        #         out =x[0]
        #         for x in out[0]:
        #             Detector.stride = torch.tensor([s/x.shape[-2]])
        #     check_anchor_order(Detector)
        #     Detector.anchors /= Detector.stride.view(-1, 1, 1)
        #     self.stride = Detector.stride
        #      # only run once
        #     # print('Strides: %s' % m.stride.tolist())    
        #     self.bias_init()  
        
        # initialize_weights(self)



            m = self.model[23]  # Detect()

            s = 128  # 2x min stride

           
            forward =  lambda x:self.forward(x)
            try:
                for x in forward(torch.zeros(1, 3, s, s)):
                    x=x[0]

                    m.stride = torch.tensor([s / x.shape[-2] ])  # forward
            except RuntimeError as e:
                if 'Not implemented on the CPU' in str(e):
                    self.model.to(torch.device('cuda'))
                    m.stride = torch.tensor([s / x.shape[-2] for x in
                    forward(torch.zeros(1, 3, s, s).to(torch.device('cuda')))])  # forward
                else:
                    raise e
            self.stride = m.stride
            Detect_DyHead.bias_init(m)  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR
 
        # Init weights, biases
        initialize_weights(self)

 
    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def net_update_temperature(self, temp):
        for m in self.modules():
            if hasattr(m, "update_temperature"):
                m.update_temperature(temp)



    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            # if i == 33:
            #     print("33 lay !")
            x = block(x)
            if i in self.seg_out_idx:     #save driving area segment result # 可驾驶区域分割和车道线检测的最后一步的激活函数为Sigmoid
                m=nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        return out
            
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module # m = Detect(  (m): ModuleList(    (0): Conv2d(128, 18, kernel_size=(1, 1), stride=(1, 1))    (1): Conv2d(256, 18, kernel_size=(1, 1), stride=(1, 1))    (2): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))  ))
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
#######################################3
    # def bias_init(self):
    #     """Initialize Detect() biases, WARNING: requires stride availability."""
    #     m = self.model[self.detector_index]  # self.model[-1]  # Detect() module
    #     # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
    #     # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
    #     for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
    #         a[-1].bias.data[:] = 1.0  # box
    #         b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

           
def get_net(cfg, **kwargs): 

    m_block_cfg = YOLOPE
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out,SAD_out = model(input_)
    detects, dring_area_seg, lane_line_seg = model_out
    Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
 