"""Job Writer.
Usage: jobWriter.py CONFIG

-h     Show this.
CONFIG Path to the config file.
"""
# Write all jobs necessary for this analysis.

import ConfigParser
import os
from os.path import basename
import glob
from docopt import docopt

def check_path(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)
        print("%s created." % folder)
    else:
        print("%s found." % folder)

def write_dagman_config(filename):
    if os.path.isfile(filename):
        print("%s found." % filename)
    else:
        F = open(filename, "w")
        print >> F, "DAGMAN_MAX_JOBS_SUBMITTED=250"
        print >> F, "DAGMAN_MAX_SUBMIT_PER_INTERVAL=50"
        print >> F, "DAGMAN_USER_LOG_SCAN_INTERVAL=10"
        print("Wrote %s" % filename)

def write_submit_file(filename, exec_path, log_path, fe_path, add):
    F = open(filename, "w")
    print >> F, "# Environment"
    print >> F, "executable = %s" % exec_path
    print >> F, "# Log + Error"
    print >> F, "initialdir = %s" % log_path
    print >> F, "output = $(initialdir)$(Jobname).$(Cluster).out"
    print >> F, "error = $(initialdir)$(Jobname).$(Cluster).err"
    print >> F, "log = %sonelog/mostRecent_$(Jobname)_condor.log" % log_path
    print >> F, "should_transfer_files = YES"
    print >> F, "when_to_transfer_output = ON_EXIT"
    print >> F, "# Notifications"
    print >> F, "notification = never"
    print >> F, "getenv = true"
    print >> F, "universe = vanilla"
    print >> F, "# Executable"
    print >> F, "Arguments = python %s $(IN_PATH) $(OUT_PATH) $(GEO) $(ID) $(NUM) %s" % (fe_path, add)
    print >> F, "queue"

    print("Wrote %s." % filename)

def write_dag_fe(filename, source, output, submit_file, geometry, id,
                 every_other=1):
    files = glob.glob(source + "*/*.bz2")
    n_files = len(files[::every_other])
    print("id: %d files" % n_files)
    
    F = open(filename, "w")
    for file in files[::every_other]:
        out_path = output + "Level4" +  basename(file)[6:]
        job_name = "feat_extr_" + basename(file)[6:-7]
	job_name = job_name.replace('.','_')
        print >> F, "JOB %s submit.sub\nVARS %s JOBNAME=\"%s\" IN_PATH=\"%s\" OUT_PATH=\"%s\" GEO=\"%s\" ID=\"%i\" NUM=\"%i\"\n" % (job_name, job_name, job_name, file, out_path, geometry, id, 1)

def write_dag_fe_multi(filename, source, output, submit_file, geometry, id):
    files = glob.glob(source + "*/*.bz2")
    n_files = len(files)
    print("id: %d files" % n_files)
    
    files = glob.glob(source + "*/")
    F = open(filename, "w")
    for file in files:
        directory = file.split("/")[-2]
        check_path(output + directory)
        out_path = output + directory + "/"
        job_name = "feat_extr_" + str(id) + "_" + directory
        print >> F, "JOB %s submit.sub\nVARS %s JOBNAME=\"%s\" IN_PATH=\"%s\" OUT_PATH=\"%s\" GEO=\"%s\" ID=\"%i\" NUM=\"%i\"\n" % (job_name, job_name, job_name, file, out_path, geometry, id, 1)

def main(config_file):
    conf = ConfigParser.ConfigParser()
    conf.read(config_file)
    
    job_path = conf.get("meta", "job_path")

    check_path(job_path + "FE_11057")
    check_path(job_path + "FE_11058")
    check_path(job_path + "FE_11499")
    check_path(job_path + "FE_44137")

    write_dagman_config(job_path + "FE_11057/dagman.config")
    write_dagman_config(job_path + "FE_11058/dagman.config")
    write_dagman_config(job_path + "FE_11499/dagman.config")
    write_dagman_config(job_path + "FE_44137/dagman.config")

    write_submit_file(job_path + "FE_11057/submit.sub",
                      conf.get("exec", "ice_rec"),
                      conf.get("out", "logs"),
                      conf.get("exec", "feature_extraction"),
                      "--sim")
    write_submit_file(job_path + "FE_11058/submit.sub",
                      conf.get("exec", "ice_rec"),
                      conf.get("out", "logs"),
                      conf.get("exec", "feature_extraction"),
                      "--sim")
    write_submit_file(job_path + "FE_11499/submit.sub",
                      conf.get("exec", "ice_rec"),
                      conf.get("out", "logs"),
                      conf.get("exec", "feature_extraction"),
                      "--sim")
    write_submit_file(job_path + "FE_44137/submit.sub",
                      conf.get("exec", "ice_rec"),
                      conf.get("out", "logs"),
                      conf.get("exec", "feature_extraction"),
                      "--sim")

    write_dag_fe(job_path + "FE_11057/job.dag",
                 	conf.get("src", "src_11057"),
                 	conf.get("out", "out_11057"),
                 	"submit.sub",
                 	conf.get("data", "geometry"),
                 	11057, every_other=int(conf.get("src", "pck_11057")))
    write_dag_fe(job_path + "FE_11058/job.dag",
                       conf.get("src", "src_11058"),
                       conf.get("out", "out_11058"),
                       "submit.sub",
                       conf.get("data", "geometry"),
                       11058, every_other=int(conf.get("src", "pck_11058")))
    write_dag_fe(job_path + "FE_11499/job.dag",
                       conf.get("src", "src_11499"),
                       conf.get("out", "out_11499"),
                       "submit.sub",
                       conf.get("data", "geometry"),
                       11499, every_other=int(conf.get("src", "pck_11499")))
    write_dag_fe(job_path + "FE_44137/job.dag",
                 conf.get("src", "src_44137"),
                 conf.get("out", "out_44137"),
                 "submit.sub",
                 conf.get("data", "geometry"),
                 44137, every_other=int(conf.get("src", "pck_44137")))


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["CONFIG"])
