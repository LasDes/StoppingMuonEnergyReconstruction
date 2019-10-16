from icecube import weighting
cache = weighting.SimprodNormalizations(filename="./dbcache.pickle")
for dataset in (11057, 11058):
    cache.refresh(dataset)
