"""Level0_preProcessing.
Usage: Level0_preProcessing.py INPUT MODEL CUT OUTPUT

-h --help  Show this screen.
INPUT      Input path     Input path to HDF5 file containing (level 4) mgs-data.
MODEL      Input path to pickle file containing models.
CUT        Confidence Cut for model prediction probability.
OUTPUT     Output directory path.
"""

from icecube import icetray, dataclasses, dataio, simclasses, gulliver, common_variables, MuonGun
from I3Tray import *
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
from collections import OrderedDict
from docopt import docopt
import glob

global cut
cut = 0.0

def passCut(frame):
    if "L5Prob" in frame:
        return frame["L5Prob"] > cut
    else:
        return False

def extract(tracklist):
    """
    Extract all features from MMCTrackList and aggregate by mean, median, min, max and variance to create input
    data for the classifier
    """
    if len(tracklist) < 1:
        return
    features = OrderedDict()
    features['x'] = []
    features['y'] = []
    features['z'] = []
    features['time'] = []
    features['zenith'] = []
    features['azimuth'] = []
    features['energy'] = []
    features['length'] = []
    features['pdg_encoding'] = []
    features['Ec'] = []
    features['Ef'] = []
    features['Ei'] = []
    features['Elost'] = []
    features['Xc'] = []
    features['Xf'] = []
    features['Xi'] = []
    features['Yc'] = []
    features['Yf'] = []
    features['Yi'] = []
    features['Zc'] = []
    features['Zf'] = []
    features['Zi'] = []
    features['Tc'] = []
    features['Tf'] = []
    features['Ti'] = []

    for track in tracklist:
        features['x'] = features['x'] + [track.xi]
        features['y'] = features['y'] + [track.yi]
        features['z'] = features['z'] + [track.zi]
        features['time'] = features['time'] + [track.particle.time]
        features['zenith'] = features['zenith'] + [track.particle.dir.zenith]
        features['azimuth'] = features['azimuth'] + [track.particle.dir.azimuth]
        features['energy'] = features['energy'] + [track.particle.energy]
        features['length'] = features['length'] + [track.particle.length]
        features['pdg_encoding'] = features['pdg_encoding'] + [float(track.particle.pdg_encoding)]
        features['Ec'] = features['Ec'] + [track.Ec]
        features['Ef'] = features['Ef'] + [track.Ef]
        features['Ei'] = features['Ei'] + [track.Ei]
        features['Elost'] = features['Elost'] + [track.Elost]
        features['Xc'] = features['Xc'] + [track.xc]
        features['Xf'] = features['Xf'] + [track.xf]
        features['Xi'] = features['Xi'] + [track.xi]
        features['Yc'] = features['Yc'] + [track.yc]
        features['Yf'] = features['Yf'] + [track.yf]
        features['Yi'] = features['Yi'] + [track.yi]
        features['Zc'] = features['Zc'] + [track.zc]
        features['Zf'] = features['Zf'] + [track.zf]
        features['Zi'] = features['Zi'] + [track.zi]
        features['Tc'] = features['Tc'] + [track.tc]
        features['Tf'] = features['Tf'] + [track.tf]
        features['Ti'] = features['Ti'] + [track.ti]

    output = np.array([])
    for key in features.keys():
        output = np.append(output, [np.mean(features[key]), np.median(features[key]), np.min(features[key]),
                                    np.max(features[key]), np.std(features[key])])

    return output

def harvest_generators(infiles):
    """
    Harvest serialized generator configurations from a set of I3 files.
    """
    generator = None
    for fname in infiles:
        f = dataio.I3File(fname)
        fr = f.pop_frame(icetray.I3Frame.Stream('S'))
        f.close()
        if fr is not None:
            for k in fr.keys():
                v = fr[k]
                if isinstance(v, MuonGun.GenerationProbability):
                    icetray.i3logging.log_info('%s: found "%s" (%s)' % (fname, k, type(v).__name__), unit="MuonGun")
                    if generator is None:
                        generator = v
                    else:
                        generator += v
    return generator

class preFilter(icetray.I3ConditionalModule):
    def __init__(self,context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("Estimator", "Model for s-classification")

    def Configure(self):
        print("Filtering Starting...")
        self._estimator = self.GetParameter("Estimator")

    def DAQ(self,frame):
        try:
            trackList = frame["MMCTrackList"]
            data = extract(trackList)
            frame["L5Prob"] = dataclasses.I3Double(self._estimator.predict_proba(data.reshape(1, -1))[0, 1])
        except:
            frame["L5Prob"] = dataclasses.I3Double(0.0)
        self.PushFrame(frame)

    def Finish(self):
        print("Finished Filtering.")


def main(arg):
    estimator = joblib.load(arg['MODEL'])
    global cut
    cut = float(arg['CUT'])
    model = MuonGun.load_model('GaisserH4a_atmod12_SIBYLL')
    generator = 1000 * harvest_generators(glob.glob(arg['INPUT']))

    tray = I3Tray()
    tray.AddModule('I3Reader','reader', FilenameList = [arg['INPUT']])
    tray.AddModule('I3MuonGun::WeightCalculatorModule', 'MuonWeight', Model=model, Generator=generator)
    #TODO find way to surpress warnings and remove failed weightings with -nan
    tray.Add(preFilter, "preFilter", Estimator=estimator)
    tray.Add("I3Writer",Filename=arg['OUTPUT'], If=passCut)
    tray.AddModule("TrashCan", "trash")
    tray.Execute()
    tray.Finish()

if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
