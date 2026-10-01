"""Build the six-domain text pool used by the CoSE experiments.

Domains: AG News, Reuters R8, TREC, MR, CR, Tweets.  Sentiment domains share the
negative/positive labels; all others own disjoint label groups (20 classes total).
"""
import csv, json, os, random, re, collections

random.seed(7)
D = os.path.join(os.path.dirname(__file__), 'data')
LABELS = ['world', 'sports', 'business', 'technology',
          'earnings', 'acquisition', 'oil', 'trade', 'currency', 'interest', 'shipping', 'grain',
          'description', 'entity', 'abbreviation', 'person', 'location', 'number',
          'negative', 'positive']
CAP_TR, CAP_TE = 2000, 500


def clean(t):
    t = t.replace('\\', ' ').replace('\n', ' ')
    return re.sub(r'\s+', ' ', t).strip()


def stratified(items, n, seed):
    rnd = random.Random(seed)
    by = collections.defaultdict(list)
    for it in items:
        by[it[1]].append(it)
    for v in by.values():
        rnd.shuffle(v)
    out, k = [], 0
    while len(out) < n and any(by.values()):
        for c in sorted(by):
            if by[c] and len(out) < n:
                out.append(by[c].pop())
    rnd.shuffle(out)
    return out


def split(items, seed):
    rnd = random.Random(seed)
    rnd.shuffle(items)
    nte = min(CAP_TE, len(items) // 5)
    return items[nte:], items[:nte]


pool = {}
# AG News
def ag(fn):
    rows = []
    with open(os.path.join(D, fn)) as f:
        for r in csv.reader(f):
            rows.append((clean(r[1] + '. ' + r[2]), int(r[0]) - 1))
    return rows
pool['ag'] = (stratified(ag('ag_train.csv'), CAP_TR, 1), stratified(ag('ag_test.csv'), CAP_TE, 2))

# Reuters R8 (single-label documents only)
r8 = {'earn': 4, 'acq': 5, 'crude': 6, 'trade': 7, 'money-fx': 8, 'interest': 9, 'ship': 10, 'grain': 11}
rt = {'training': [], 'test': []}
for line in open(os.path.join(D, 'reuters', 'cats.txt')):
    p = line.split()
    if len(p) != 2 or p[1] not in r8:
        continue
    sp = p[0].split('/')[0]
    txt = clean(open(os.path.join(D, 'reuters', p[0]), encoding='latin-1').read())
    rt[sp].append((txt, r8[p[1]]))
def capped(items, cap_per, seed):
    rnd = random.Random(seed); by = collections.defaultdict(list)
    for it in items: by[it[1]].append(it)
    out = []
    for c, v in by.items():
        rnd.shuffle(v); out += v[:cap_per]
    rnd.shuffle(out); return out
pool['r8'] = (capped(rt['training'], 400, 3)[:CAP_TR], capped(rt['test'], 100, 4)[:CAP_TE])

# TREC coarse (DESC, ENTY, ABBR, HUM, LOC, NUM)
def kim(fn, off):
    out = []
    for line in open(os.path.join(D, fn), encoding='latin-1'):
        p = line.strip().split(' ', 1)
        if len(p) < 2: continue
        lab, txt = p
        out.append((clean(txt), off + int(lab)))
    return out
tr = kim('TREC.train.all', 12); random.Random(5).shuffle(tr)
pool['trec'] = (tr[:CAP_TR], kim('TREC.test.all', 12)[:CAP_TE])
pool['mr'] = split(kim('rt-polarity.all', 18), 6)
pool['cr'] = split(kim('custrev.all', 18), 8)
tw = []
for fn, lab in [('negative_tweets.json', 18), ('positive_tweets.json', 19)]:
    for line in open(os.path.join(D, 'twitter_samples', fn)):
        tw.append((clean(json.loads(line)['text']), lab))
pool['tweet'] = split(tw, 9)
for k in ['mr', 'cr', 'tweet']:
    trn, te = pool[k]
    pool[k] = (stratified(trn, CAP_TR, 10), stratified(te, CAP_TE, 11))

with open(os.path.join(D, 'pool.jsonl'), 'w') as f:
    for dom, (trn, te) in pool.items():
        for sp, items in [('train', trn), ('test', te)]:
            for t, y in items:
                f.write(json.dumps({'domain': dom, 'split': sp, 'text': t, 'label': y}) + '\n')
        print(dom, len(trn), len(te), collections.Counter(y for _, y in trn).most_common(3))
json.dump(LABELS, open(os.path.join(D, 'labels.json'), 'w'))

# External test set: MPQA opinion phrases (never streamed by any node)
rows = []
for line in open(os.path.join(D, 'mpqa.all'), encoding='latin-1'):
    p = line.strip().split(' ', 1)
    if len(p) < 2:
        continue
    rows.append({'domain': 'mpqa', 'split': 'test', 'text': clean(p[1]), 'label': 18 + int(p[0])})
random.Random(3).shuffle(rows)
neg = [r for r in rows if r['label'] == 18][:500]; pos = [r for r in rows if r['label'] == 19][:500]
with open(os.path.join(D, 'pool_ext.jsonl'), 'w') as f:
    for r in neg + pos:
        f.write(json.dumps(r) + '\n')
print('mpqa', len(neg), len(pos))
