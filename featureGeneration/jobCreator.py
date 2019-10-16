from glob import glob
from os.path import basename

# Constants
SIM_NUM = 11058
DATA_DIR = "/data/ana/Muon/level3/sim/2012/CORSIKA-in-ice/%i/" % SIM_NUM
GEO_DIR = "/data/user/thoinka/det.i3.bz2"
MAX_ID = 31999
OUTPUT_DIR = "/data/user/thoinka/%i_level4/" % SIM_NUM

JOB_PREFIX = "fextraction_"
JOB_OUTPUT = "/home/thoinka/jobs/fextraction/FE_all.dag"


# First go into all folders and check those out.
comp_path = DATA_DIR + "*/*.bz2"
all_bz2 = glob(comp_path)
n_files = len(all_bz2)

# Now write Jobs for everything
output_file = open(JOB_OUTPUT, "w")
for f in all_bz2:
	job_name = JOB_PREFIX + basename(f)[:-7]
	output_file.write("JOB %s submit.sub\nVARS %s JOBNAME=\"%s\" INPUT=\"%s\" OUTPUT=\"%s\" GEO=\"%s\" ID=\"%i\" NUM=\"%i\"\n" % (job_name, job_name, job_name, f, OUTPUT_DIR + "Level4_" + basename(f)[7:], GEO_DIR, SIM_NUM, n_files))