# Apriori Algorithm
# 实现对事物集D中频繁项集(frequent itemsets)的查询
# 更新时间：2019/3/19(更新至朴素Apriori的子方法完成)

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
        """D is a list of lists while each sub-list is 
        a transaction"""
        self.threshold = counts_threshold
        self.D = D

    def _apriori_gen(self,former_L):
        """
            private_function:
            used the L(k-1) to form a C(k)
            input: former_L  a list of tuples
        """
        lformer_L = []
        for x in former_L:
            lformer_L.append(x[0])

        Ck = []
        former_Llen = len(lformer_L)
        k = len(former_L[0])+1
        for subi in range(0,former_Llen):
            for subj in range(subi+1,former_Llen):
                if subi[:k-2] == subj[:k-2]:
                    c = subi.append(subj[-1])
                    if not _has_infrequent_subset(c,lformer_L):
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
            temp = c[:i]+c[i:]
            if temp not in lformer_L:
                return True
        return False
