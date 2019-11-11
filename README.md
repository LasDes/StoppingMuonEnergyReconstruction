# StoppingMuonEnergyReconstruction
Analysis chain used for the reconstruction of the surface energy spectrum of atmospheric muons on IceCube MonteCarlo
simulations. The analysis is divided into multiple steps, containing:

* Routines to extract a stopping muon sample from IceCube Level 3 data using machine learning algorithms provided by sklearn.
* Scripts to calculate IceCube's effective area for the detection of single stopping muons

## Dependencies
The routines and methods provided here are based on Python 2.7.13. Additionally,
the following python modules are required:

* docopt (0.6.2)
* h5py (2.8.0)
* numpy (1.14.0)
* pandas (0.23.4)
* scikit-learn (0.20.1)
* scipy (0.19.0)

## Extraction of a Stopping Muon Sample
Extracting a sample requires multiple steps starting with Muon Level 3 data.
First, the Level 3 data needs to be processed to Level 4 data. This can be
done using the `feature_extraction.py` routine

```
>>> python featureGeneration/featureExtraction.py -i <input> -o <output> -g <geometry> [--sim --multi]
```

wheres `input` refers to the Level 3 input file `output` refers to the output
Level 4 file, `geometry` needs to be a i3 file containing a geometry frame and
the option `--sim` should be used when processing simulated data. The option
`--multi` can be used to process multiple input files simultaneously. The input
can than be given in a `glob`-readable format. This can save a lot of time when
dealing with low-energy input.

After the data is processed, it needs to be stored in hdf5 files that are readable
for the following machine learning procedures. This is done using the
`hdf5_writer.py` routine

```
>>> python featureGeneration/hdf5_writer.py -i <input> -o <output> [--exp]
```
The `input` should be a `glob`-readable path, the output a single hdf5-file.
The option `--exp` should be used when dealing with recorded data to avoid
the writing of attributes relating to the MC truth.

If no model is provided, the model can be trained using simulated data, using
```
>>> python estimatorModelTraining/feature_selection.py --input_mc <input_mc> --input_data <input_data>
-o <output> --n_mismatch <n_mismatch> -j <n_jobs> --n_estimators <n_estimators>
```
whereas `input_mc` is the path to all simulated data hdf5-files (if more than
one file is needed, separate them with commas, such as `path1,path2,...`).
`input_data` is the path to recorded data hdf5-file. `output` is a pickle-file
containing the results of the feature selection. The rest are hyperparameters
with the defaults

* `--n_mistmach` 120
* `--n_estimators` 100

Next, the model itself is trained, using

```
>>> python estimatorModelTraining/train_model.py -i <input> -f <fs_output> -j <n_jobs> -o <output>
-s <subsamplefraction> [--n_estimators <n_estimators --min_samples <min_samples>
```
with `input` being the path to the hdf5 files corresponding to simulated data,
again separated by commas if more than one is given, `fs_output` the output of
the feature selection routine. The `output` should be a path to a directory
where the files `cv_predictions.pickle` and `models.pickle` are stored, the
first containing results of the cross validation, the second containing the
trained models, both in a dictionary. The rest are hyperparameters with the
default values

* `--n_estimators 100`
* `--min_samples 8`
* `--s 1.0`

After the model is generated, it can be used with the `apply_model.py`-routine
```
>>> python estimatorModelTraining/apply_model.py -i <input> -m <model> -o <output>
```
Whereas the `input` is a hdf5-file retrieved from recorded data, `model` is a
`models.pickle` file from before, and `output` will be a csv-file with the
resulting scores.

The model contains four different estimators:

* a random forest classifier for selection of stopping muon events
* a random forest classifier for selection of single stopping muon events
* a random forest regressor for estimation of angular reconstruction quality for an event
* a random forest regressor for estimation of stopping depth

which estimate the s-score, m-score, q-score and r-score respectively for each event.
Level 4 data that has been outfitted with these scores is regarded as Level 5 data.

By introducing appropriate cuts for m- and q-score, a highly pure set of single stopping
muon events with good angular reconstructions can be selected.

## Estimation of the Effective Area

This step requires IceCube Level 3 MonteCarlo data for Muons, both from CORSIKA and MuonGun simulations.
The CORSIKA data is available in IceCubes data-warehouse, while the MuonGun data has to ge generated using 'simulation_scripts',
available under:
```
https://github.com/mbrner/simulation_scripts
```
The software only runs on the NFX cluster of UWC Madison and is called via
```
>>> python simulation_scripts.py -s "step" "path_to_config_yaml"
```
where "path_to_config_yaml" is to be replaced with "configs/muongun_static.yaml". The script must be executed
for steps 0 to 5 successively.
The simulation parameters in this config are set to:

* `generator: muongun`
* `e_min: 10`
* `e_max: !!float 1e5`
* `gamma: 3`
* `icemodel: SpiceLea`
* `muongun_e_break: !!float 1e6`
* `muongun_generator: static`
* `muongun_model: GaisserH4a_atmod12_SIBYLL`
* `muongun_min_multiplicity: 1`
* `muongun_max_multiplicity: 1`

but may be altered.
The final output of step 5 is Muon Level 3 data. Thereafter, the results of step
1 and 5 undergo a feature generation and are writen to hdf5-files, similar to above.

The effective area for single stopping muons is calculated by calling
```
>>> effectiveAreaEstimation/effectiveAreaCalculator.py PRE POST MC MODELS OUTPUT EMIN EMAX ZENMAX EBINS ZENBINS BATCHES [--read_pickle]
```
Here 'PRE' refers to the hdf5-file containing the MuonGun results after step 1, while 'POST' refers to the hdf5-file
contaning the MuonGun results after step 5. 'MC' is the hdf5-file containing CORSIKA Level-3 MC-data. 'MODELS' denotes the
path to the pickle-file containing the estimator-models, while 'OUTPUT' is the directory where results are to be stored.
'EMIN' and 'EMAX' signify the range of energies. The range of zenith angles goes from 0 to 'ZENMAX'. The number of bins for
zenith and energy are given by 'ZENBINS' and 'EBINS' respectively. The input data is can be read in chunks, should the
RAM-limitations require this. The number of chunks is given by 'BATCHES'.
The script writes it's results into the output directory as 'effArea_mgs_corsica_total', both as csv and pickle.
Additionally some intermediate results from processing the CORSIKA data are dumped into 'df_corsica.pickle' and
'df_corsica_est.pickle'. These can be reused, when rerunning the script, by setting the '--read_pickle' flag,
which reduces runtime significantly.
Analogously, the effective area for all Muons can be estimated through
```
>>> effectiveAreaEstimation/effectiveAreaCalculator_nonstop.py PRE POST MC MODELS OUTPUT EMIN EMAX ZENMAX EBINS ZENBINS BATCHES [--read_pickle]
```

## Unfolding of the Muon Spectrum

The unfolding process is performed within two jupyter notebooks: 'jupyterNotebooks/unfolding_final.ipynb' and
'jupyterNotebooks/PlottingResults.ipynb', which have to be executed successively. Beforehand the path variable
'data_dir' in line 3 or 7 respectively has to be adapted, so that it points to a directory which contains the files

* `df_corsica_est.pickle`
* `effArea_mgs_corsica_total.pickle`
* `effArea_mgs_corsica_nonstop_total.pickle`

These are the output of the effective area estimation step above.

## Submitting Condor-Jobs generated with JobWriters

Due to the large amount of data likely to be involved in this analysis, it is recommended to perform much, if not all, of the
steps described above on a remote server cluster that offers far greater computational capacities than any local machine.
The Icecube collaboration has access to the NFX-Cluster at UW Madison.
The so called COLBAT nodes can be used interactivley like the local machine and are primarily used for tests and less expensive
processing steps that require only a limited amount of CPU cores and memory.
On the other hand, processing thousands of individual Level-3 MonteCarlo files during the featureExtraction step is best done using
the CONDOR system. Since the MonteCarlo files are processed indipendently, the featureExtraction-step can be split into
jobs that handle only one single file. These jobs can be submited to the CONDOR system with then manages their execution. This approach
will probably be far less time consuming than featureExtraction for the whole set on COBALT.
The jobWriter programs are used to generate scripts to instruct CONDOR. Executing

```
>>> jobWriter/jobWriter.py jobs.config
```

on any node of the NFX-Cluster (CONDOR or submitter) will create four directories.

* `/home/sninfa/jobs/FE_11057`
* `/home/sninfa/jobs/FE_11058`
* `/home/sninfa/jobs/FE_11499`
* `/home/sninfa/jobs/FE_44137`

These target directories can be altered by editing the jobs.config file. The directories initially contain three files each.

* `dagman.config`
* `job.dag`
* `submit.sub`

The simultaneous submission of all jobs in directory X is achieved by the entering the command line

```
>>> condor_submit_dag -config dagman.config job_hdf5.dag
```

from the submitter node. The same command has to be entered from each of the four directories. (Actually the fourth directory
can be omitted, since the 44137 dataset is not used and should not be used!)
For each of the processed Level-3 MonteCarlo files a Level-4 file is created and stored under the respective output directory.

* `/data/user/sninfa/level4_mc/11057`
* `/data/user/sninfa/level4_mc/11058`
* `/data/user/sninfa/level4_mc/11499`
* `/data/user/sninfa/level4_mc/44137`

The newly created Level-4 files are then cast together into one large .hdf5 file using hdf5_writer.py. This process could
theoretically also be submitted to CONDOR but since this task is not parallelizable and only occupies one CPU, it can be run
on COBALT, which is the more effective approach here.

The results from the MuonGun simulation (mgs) are processed in a similar manner. The submission scripts are generated through

```
>>> jobWriter/jobWriter_mgs.py jobs_mgs.config
>>> jobWriter/jobWriter_mgs_labeling.py jobs_mgs_labeling.config
```

where mgs is applied to the output of the simulation chain after step 5 and mgs_labeling is applied to the output after step 1.

All other jobWriters can be safely ignored as they are not integrated into the final analysis chain.
