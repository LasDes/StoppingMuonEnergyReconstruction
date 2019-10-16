# # # # # # # # #
# Config Parser #
# # # # # # # # #
#
# Method to parse a config file and turn it into a routine
# 
# 2017 T. Hoinka (tobias.hoinka@udo.edu)

import ConfigParser
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.pipeline import make_pipeline

def remove_clutter(string):
	return string.replace("\n", "").replace(" ", "").replace("\t", "")

def prs_i(entry):
	if entry == "None":
		return None
	else:
		return int(entry)

def prs_f(entry):
	if entry == "None":
		return None
	else:
		return float(entry)

def push_feature_importances(x, y, feature_importances=None):
	return feature_importances, feature_importances

def config_parser(config_file):
	conf = ConfigParser.ConfigParser()
	conf.optionxform = str
	conf.read(config_file)
	conf_dict = {}
	conf_dict["data"] = {}
	data_list = remove_clutter(conf.get("data", "data_list")).split(",")
	sample_fraction = float(conf.get("data", "sample_fraction"))
	conf_dict["data"]["data_list"] = data_list
	conf_dict["data"]["sample_fraction"] = sample_fraction
	
	conf_dict["pipeline_s"] = {}
	for name, content in conf.items("pipeline_s"):
		conf_dict["pipeline_s"][name] = content
	conf_dict["pipeline_s"]["plots"] = remove_clutter(conf_dict["pipeline_s"]["plots"]).split(",")

	conf_dict["pipeline_q"] = {}
	for name, content in conf.items("pipeline_q"):
		conf_dict["pipeline_q"][name] = content
	conf_dict["pipeline_q"]["plots"] = remove_clutter(conf_dict["pipeline_q"]["plots"]).split(",")

	conf_dict["pipeline_m"] = {}
	for name, content in conf.items("pipeline_m"):
		conf_dict["pipeline_m"][name] = content
	conf_dict["pipeline_m"]["plots"] = remove_clutter(conf_dict["pipeline_m"]["plots"]).split(",")


	conf_dict["pipeline_r"] = {}
	for name, content in conf.items("pipeline_r"):
		conf_dict["pipeline_r"][name] = content
	conf_dict["pipeline_r"]["plots"] = remove_clutter(conf_dict["pipeline_r"]["plots"]).split(",")

	conf_dict["output"] = {}
	for name, content in conf.items("output"):
		conf_dict["output"][name] = content

	conf_dict["misc"] = {}
	for name, content in conf.items("misc"):
		conf_dict["misc"][name] = content

	return conf_dict