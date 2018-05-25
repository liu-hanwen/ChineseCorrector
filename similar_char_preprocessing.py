import CharSimilarity
import pandas as pd
import numpy as np
import pickle
from multiprocessing import Pool
import os, time, random
import math

POOL_SIZE = 4

simi_paras = (True, True, 0.6, 0.1, 0.2, 0.1)
ch_list = list(CharSimilarity.sijiao_dict.dic.keys())

ssc_dic = {}

start = time.time()

def ssc_(ch):
    return ch,CharSimilarity.ssc(ch)

with Pool(4) as p:
    for ch,ssc_value in p.map(ssc_, ch_list[:]):
        ssc_dic[ch] = ssc_value

end = time.time()

print('ssc_task time: ', end-start, 'sec')

def simi_(chs):
    return chs, CharSimilarity.similarity(*chs,*simi_paras)

start = time.time()

mapper_list = []

for i in range(len(ch_list)-1):
    for j in range(i+1, len(ch_list)):
        ch1, ch2 = ch_list[i], ch_list[j]
        mapper_list.append((ch1, ch2))

simi_dic = {}

with Pool(4) as p:
    for ch_tuple, simi_value in p.map(simi_, mapper_list[:]):
        ch1 = ch_tuple[0]
        ch2 = ch_tuple[1]

        if ch1 not in simi_dic:
            simi_dic[ch1] = {}
        if ch2 not in simi_dic:
            simi_dic[ch2] = {}

        simi_dic[ch1][ch2] = simi_value
        simi_dic[ch2][ch1] = simi_value

end = time.time()

print('simi_task time: ', end-start, 'sec')

start = time.time()

def dic2pdS(tp):
    key = tp[0]
    item = tp[1]
    return key, pd.Series(item)

def dic_iter(tmp_dic):
    for key in tmp_dic:
        yield (key, tmp_dic[key])


with Pool(4) as p:
    for key, item in p.map(dic2pdS, dic_iter(simi_dic)):
        simi_dic[key] = item

end = time.time()

print('2pdSeries_task time: ', end-start, 'sec')

print(simi_dic['的'].nlargest(5))

with open('./ssc_cache/%s.pkl' % str(simi_paras), 'wb') as f:
    pickle.dump(simi_dic, f)

print('done')