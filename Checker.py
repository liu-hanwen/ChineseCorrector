'''
Author: LIU Hanwen
Email: liu.hanwen@foxmail.com
Github: https://github.com/liu-hanwen/
Created Date: April 28, 2018
'''
import pandas as pd
import pickle
from multiprocessing import Pool, Pipe
import jieba
import CharSimilarity
import kenlm
import itertools

'''参数'''
SIMI_DIC_PATH = './pd_simi_dic.pkl' # 相近字字典文件路径
LANG_MODEL_PATH = '../data/weibo_contents_words.bin' # 语言模型文件路径
VOCAB_PATH = '../data/weibo_contents_words.set'
MAX_MAYBE_WRONG_SIZE = 6 # 错字窗口最大值
N_GRAM = 3
SIMI_THRESHOLD = 0.5 # 超过阈值的相近字就会被匹配
TOP_N_CANDIDATE = 3

'''常见中文字表'''
ch_list = list(CharSimilarity.sijiao_dict.dic.keys())

'''过滤字表'''
ignore_list = "的你我了就"

'''加载形近值字表'''
simi_dic = None
with open(SIMI_DIC_PATH, 'rb') as f:
    simi_dic = pickle.load(f)

'''加载词汇表'''
vocab_dic = None
with open(VOCAB_PATH, 'rb') as f:
    vocab_dic = pickle.load(f)

'''加载语言模型'''
lang_model = kenlm.Model(LANG_MODEL_PATH)

print('PREPROCESSING DONE!')

def substr(text):
    if len(text)==1:
        yield text
    else:
        yield [''.join(text)]
        for i in range(1, len(text)):
            left = [''.join(text[:i])]
            right = text[i:]
            for sub in substr(right):
                yield left + sub

def seek4simi(word):

    l = len(word)
    simi_value = 0.0
    not_in_dic_flag = False
    
    for fixed_word in vocab_dic[l]:
        
        for i in range(l):
            try:
                if word[i]==fixed_word[i]:
                    simi_value+=1.0
                else:
                    simi_value += simi_dic[word[i]][fixed_word[i]]
            except KeyError:
                not_in_dic_flag = True
                break
        if not not_in_dic_flag:
            
            if simi_value / l > SIMI_THRESHOLD:
                yield fixed_word, simi_value/l
            else:
                pass
            simi_value = 0.0
        else:
            not_in_dic_flag = False
    return
   

def correct_algo(singletons, text):
    if len(singletons)==1:
        return text[singletons[0]]
    maybe_wrong = [text[idx] for idx in singletons]
    prefix = [text[idx] for idx in range(singletons[0]-N_GRAM+1, singletons[0]) if idx>=0]
    boost_dic = {}
    candidates = list(substr(maybe_wrong))
    ret = {}

    for candidate in candidates:
        for wrong_word in candidate:
            if wrong_word not in boost_dic:
                boost_dic[wrong_word] = {}
                for fixed_word, simi_value in seek4simi(wrong_word):
                    boost_dic[wrong_word][fixed_word] = simi_value
                boost_dic[wrong_word] = pd.Series(boost_dic[wrong_word])
            else:
                pass

        fixed_products = itertools.product(*tuple(boost_dic[wrong_word].nlargest(TOP_N_CANDIDATE).index.tolist() for wrong_word in candidate))

        for fixed_product in fixed_products:
            ret[''.join(list(fixed_product))] = lang_model.score(' '.join(prefix + [''.join(list(fixed_product))]))

    ret = pd.Series(ret)
    return ret.nlargest(TOP_N_CANDIDATE).index.tolist()
        

def correct_core(text):
    singletons = [] # Storing singletons index
    text_splited = list(jieba.cut(text+'。', HMM = False))
    for word_index, word in enumerate(text_splited):
        if len(word)==1: # Singleton?
            if word in ignore_list or word not in ch_list or len(singletons)>MAX_MAYBE_WRONG_SIZE: # Frequent?
                if len(singletons)!=0:
                    newWord = correct_algo(singletons, text_splited)
                    yield newWord
                if word_index!=len(text_splited) - 1:
                    yield word
                singletons = []
            else:
                singletons.append(word_index)
        else:
            if len(singletons)!=0:
                newWord = correct_algo(singletons, text_splited)
                yield newWord
            if word_index!=len(text_splited) - 1:
                yield word
            singletons = []