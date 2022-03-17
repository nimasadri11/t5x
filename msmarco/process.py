from collections import defaultdict
import sys

with open('docs.tsv') as f:
    lines = f.readlines()

qs = set()
rels = defaultdict(set)
not_rels = defaultdict(set)

for line in lines:
#    if len(qs) > 99:
#        sys.exit()
    line = ''.join(line.split('\n')[:-1])
    query, pos, neg = line.split('\t')
#    if query not in qs:
#        print("^" + query)
    qs.add(query)
    rels[query].add(pos)
    not_rels[query].add(neg)

for q in qs:
    pos_set = rels[q]
    neg_set = not_rels[q]
    for pos in pos_set:
        print(q, '\t', pos, '\t', 1)
    for neg in neg_set:
        print(q, '\t', neg, '\t', 0)

