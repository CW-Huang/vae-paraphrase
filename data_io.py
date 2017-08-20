import random
import cPickle as pickle
import threading
import Queue
import numpy as np


def stream(data_file, word2idx):
    unk_idx = len(word2idx)
    for line in open(data_file):
        lines = line.strip().split('\t')
        line_idxs = []
        for line in lines:
            idxs = np.array([word2idx.get(w, unk_idx) for w in line.split()],
                            dtype=np.int16)
            line_idxs.append(idxs)
        if len(line_idxs) == 2:
            yield tuple(line_idxs)


def async(stream, queue_size=2):
    queue = Queue.Queue(maxsize=queue_size)
    end_marker = object()

    def producer():
        for item in stream:
            queue.put(item)
        queue.put(end_marker)

    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()
    # run as consumer
    item = queue.get()
    while item is not end_marker:
        yield item
        queue.task_done()
        item = queue.get()


def randomise(stream, buffer_size=100):
    buf = buffer_size * [None]
    ptr = 0
    for item in stream:
        buf[ptr] = item
        ptr += 1
        if ptr == buffer_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            ptr = 0
    buf = buf[:ptr]
    random.shuffle(buf)
    for x in buf:
        yield x


def sortify(stream, key, buffer_size=200):
    buf = buffer_size * [None]
    ptr = 0
    for item in stream:
        buf[ptr] = item
        ptr += 1
        if ptr == buffer_size:
            buf.sort(key=key)
            for x in buf:
                yield x
            ptr = 0
    buf = buf[:ptr]
    buf.sort(key=key)
    for x in buf:
        yield x


def batch(stream, batch_size=10):
    batch = []
    for item in stream:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def arrayify(stream, start_idx, end_idx):
    for batch in stream:
        batch_tuple = []
        for pair in xrange(2):
            batch_idxs = np.zeros((
                len(batch),
                max(f[pair].shape[0] for f in batch) + 2,
            ), dtype=np.int32) - 1
            batch_idxs[:, 0] = start_idx
            for i, s in enumerate(batch):
                batch_idxs[i, 1:s[pair].shape[0] + 1] = s[pair]
                batch_idxs[i, s[pair].shape[0] + 1] = end_idx
            batch_tuple.append(batch_idxs)
        yield tuple(batch_tuple)

def arrayify2(stream, start_idx, end_idx):
    for batch_ in stream:
        batch_tuple = []
        batch = [f[0] for f in batch_] + [f[1] for f in batch_]
        batch_idxs = np.zeros((
            len(batch),
            max(f.shape[0] for f in batch) + 2,
        ), dtype=np.int32) - 1
        batch_idxs[:, 0] = start_idx
        for i, s in enumerate(batch):
            batch_idxs[i, 1:s.shape[0] + 1] = s
            batch_idxs[i, s.shape[0] + 1] = end_idx
        batch_tuple.append(batch_idxs)
        yield tuple(batch_tuple)



def load_dictionary(dict_file):
    idx2word = pickle.load(open(dict_file, 'rb'))
    word2idx = {w: i for i, w in enumerate(idx2word)}
    idx2word += ['<unk>']
    return idx2word, word2idx


if __name__ == "__main__":
    from collections import Counter

    tokens = Counter(w for line in open('/data/lisa/data/sheny/ParaNews/train.txt')
                     for w in line.strip().split())
    dictionary = [w for w, c in tokens.most_common(10000)
                  if c > 1]
    dictionary.sort()
    pickle.dump(dictionary, open('dict.pkl', 'wb'))
