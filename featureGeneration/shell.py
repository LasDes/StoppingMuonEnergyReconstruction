F = open("make_all.sh", "w")

for i in range(1000):
	F.write("python featureExtraction.py ../data11058/Level3_IC86.2012_corsika.011058.%06d.i3.bz2 ../11058_level4/Level4_IC86.2012_corsika.011058.%06d.i3.bz2 ../Detector.i3.gz 11058 997 ./dbcache.pickle --sim\n" % (i, i))