# Apriori Algorithm
# 实现对事物集D中频繁项集(frequent itemsets)的查询
# 更新时间：2019/3/19(更新至朴素Apriori的子方法完成)
#          2019/3/20(实现朴素Apriori，先验知识尚未使用/Hash-Table优化尚未使用)
#          2019/3/21(先验知识子函数完成并验证)
#          2019/3/31(Debug，再次验证完毕)

from functools import reduce
from collections import Counter
from copy import deepcopy
import time

class Apriori:
    """
        take the transaction-set D and 
        analysis its frequent itemsets

        input: list of lists D and counts_threshold
               which means minimum (absolute) 
               support threshold
        output: a list of tuples whose struct is like
                ([transaction],support) 

    """
    def __init__(self,D,counts_threshold):
        """
            D is a list of lists while each sub-list is 
            a transaction
        """
        self.min_sup = counts_threshold
        # each item in every transactions should be sorted
        self.D = D

    def _apriori_gen(self,former_L,k):
        """
            private_function:
            used the L(k-1) to form a C(k)
            input: former_L  /a list of tuples
                   k         /parameter k for C(k)
        """
        # itemset list of the C[k-1]: don't need
        # 
        lformer_L = [x[0] for x in former_L]

        # A candidate result Ck:list of lists
        Ck = []
        former_Llen = len(lformer_L)
        # Need to be optimized
        # and these operations need sorted first
        for subi in range(0,former_Llen):
            for subj in range(subi+1,former_Llen):
                if (k>1 and lformer_L[subi][:(k-1)] == lformer_L[subj][:(k-1)]):
                    c = deepcopy(lformer_L[subi])
                    c.append(lformer_L[subj][-1])
                    # 'c' is a candidate item and
                    # the key point:we need to keep the orders
                    c = sorted(c)
                    if not self._has_infrequent_subset(c,lformer_L):
                        Ck.append(c)
                elif k-2<0:
                    c = lformer_L[subi]+lformer_L[subj]
                    if not self._has_infrequent_subset(c,lformer_L):
                        Ck.append(c)
        return Ck

    def _has_infrequent_subset(self,c,lformer_L):
        """
            private_function:
            to ensure whether there is a infrequent_subset in c
            input: c(the candidate K itemsets) 
                   lformer_L(the list of K-1 itemsets)
        """
        len_c = len(c)             
        for i in range(0,len_c):
            temp = c[:i]+c[i+1:]
            if temp not in lformer_L:
                return True
        return False

    def _find_frequent1(self):
        """
            private_function:
            use the self.D to build the L1
        """
        count = Counter(reduce(lambda x,y:x+y,self.D))
        C1 = [list(x) for x in list(count.items())]
        L1 = []
        for x in C1:
            x[0] = [x[0]]
            if (x[1] >= self.min_sup):
                L1.append(x)
        return [tuple(x) for x in L1]
        

    def find_frequent(self):
        """
            public_function:
            Apply Apriori to find frequent itemsets.
        """
        D_set = [set(x) for x in self.D]
        D_len = len(self.D)

        L=[]
        L.append(self._find_frequent1())
        FreItem_nums = len(L[0])
        k = 0
        while(L[k] != [] and k < FreItem_nums):
            k += 1
            Candidate = self._apriori_gen(L[k-1],k)
            Candidate_len = len(Candidate)
            pattern_counts = [0 for x in range(0,Candidate_len)]
            # use 'set' to make sure if the items are in D
            Candidate_set = [set(x) for x in Candidate]
            for i in  range(0,Candidate_len):
                for j in range(0,D_len):
                    if(Candidate_set[i].issubset(D_set[j])):
                        pattern_counts[i] += 1
            
            temp = list(zip(Candidate, pattern_counts))
            # Select just the itemsets whose support >= min_sup
            L.append(list(filter(lambda x:x[1]>=self.min_sup, temp)))
        
        return reduce(lambda x,y:x+y,L)

# test #

if __name__ == '__main__':
    D=[['I1','I2','I5'],
       ['I2','I4'],
       ['I2','I3'],
       ['I1','I2','I4'],
       ['I1','I3'],
       ['I2','I3'],
       ['I1','I3'],
       ['I1','I2','I3','I5'],
       ['I1','I2','I3']]
    # D=[['M','O','N','K','E','Y'],
    #    ['K','O','N','K','E','Y'],
    #    ['M','A','K','E'],
    #    ['M','U','C','K','Y'],
    #    ['C','O','K','I','E']]

    apriori = Apriori(D, 2)
    start_time = time.time()
    L = apriori.find_frequent()
    print(L)
    print(len(L))
    print("Cost time: %.3fs"%(time.time()-start_time))