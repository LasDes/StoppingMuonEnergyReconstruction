from __future__ import division
from __future__ import print_function
from os import walk
from os.path import join
from os import listdir

jobname = "FExtraction"
indir = "/data/ana/Muon/level3/sim/2012/CORSIKA-in-ice/11058/00000-00999/" 
outdir = "/data/user/thoinka/11058_L4/"
t = "nue"

for file in listdir(indir):
	# filter out only run ID"s ending with 0
	# b.c. of blindness
	print("JOB %s submit.sub" % (jobname+file[:-7]))
	print("VARS %s JOBNAME=\"%s\" inputfile=\"%s\" outputpath=\"%s\" type=\"%s\"" % (jobname+file[:-7], jobname+file[:-7], indir+"/"+file, outdir, t))
