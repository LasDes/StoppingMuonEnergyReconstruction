import numpy as np

def even_slice_(N, n_slices):
	p = []
	if N == 0:
		return np.array(p)
	if N < n_slices:
		return np.random.randint(n_slices, size=N)
	fraction_N = int(N / n_slices)
	for i in range(n_slices):
		p += [np.full(fraction_N, i, dtype=int)]
	rest = N - n_slices * fraction_N
	if rest > 0:
		p += [np.random.randint(n_slices, size=N - n_slices * fraction_N)]
	p = np.concatenate(p)
	np.random.shuffle(p)
	return p

"""Generates cross validation slices stratified over both groups and labels.
No generator bullshit.

Parameters
----------
label : array, shape = [n_samples,]
        Labels.

group : array, shape = [n_samples,]
        Group labels.
        
n_slices : integer, optional
           Number of slices to generate.

Returns
-------
indices : list(tuples)
          A list containing tuples like (train_idx, test_idx) each containing
          boolean masks with shape [n_samples,].
"""
def group_cv_slices(label, group, n_slices=10):
	group_set = list(set(group))
	label_set = list(set(label))
	n_groups = len(group_set)
	perm = np.zeros(len(label))
	for g in group_set:
		for l in label_set:
			mask = (group == g) & (label == l)
			perm[(group == g) & (label == l)] = even_slice_(np.sum(mask),
				                                            n_slices)
	test_idx = []
	train_idx = []
	for i in range(n_slices):
		test_idx += [perm == i]
		train_idx += [perm != i]
	return zip(train_idx, test_idx)