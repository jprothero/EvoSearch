
# coding: utf-8

# In[ ]:


from functools import reduce
import itertools
import operator
import numpy as np

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def sorted_factors(n):    
    factors = list(set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))
    factors.sort()
    return factors

def roundUp(numToRound, multiple):
    if (multiple == 0):
        return numToRound

    remainder = numToRound % multiple
    if (remainder == 0):
        return numToRound

    return numToRound + multiple - remainder

def middle_factors(n):
    factors = sorted_factors(n)
    if len(factors) % 2 == 1:
        factors = [
            factors[int(np.floor(len(factors)/2))],
            factors[int(np.floor(len(factors)/2))]
        ]
        return factors
    left_middle = int(np.floor(len(factors)/2)) - 1
    right_middle = int(np.ceil(len(factors)/2))

    if left_middle == right_middle - 1:
        right_middle += 1
    factors = factors[left_middle:right_middle]
    
    return factors

def convert_to_squarish_matrix(M):
    target_vector_size = int(roundUp((M.shape[1] * M.shape[0]) / 2, 2))
    factors = get_middle_factors(target_vector_size)

    target_matrix = np.zeros((factors[0], factors[1]))

    target_i = 0
    target_j = 0

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if j > i:
                target_matrix[target_i, target_j] = A[i, j]
                target_j = (target_j + 1) % factors[1]
                if target_j == 0:
                    target_i += 1

    return target_matrix

def most_common_factors(L):
    L.sort()
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
            # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    # pick the highest-count/earliest item
    most_common = max(groups, key=_auxfun)[0]
    mc_idx = L.index(most_common)
    comp_idx = min(max(-mc_idx - 1, -len(L)), -1)
    print(comp_idx)
    component_factor = L[comp_idx]
    return most_common, component_factor

