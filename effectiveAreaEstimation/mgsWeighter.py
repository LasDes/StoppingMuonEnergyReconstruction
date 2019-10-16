"""mgsWeighter.
Usage: mgsWeighter.py INPUT OUTPUT

-h --help  Show this screen.
INPUT       Input path.
OUTPUT      Output path.
"""

from icecube import icetray, dataclasses, dataio, simclasses, gulliver, common_variables, MuonGun
from I3Tray import *
from docopt import docopt
import glob

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

def main(arg):
    model = MuonGun.load_model('GaisserH4a_atmod12_SIBYLL')
    generator = 10000 * harvest_generators(glob.glob(arg['INPUT']))

    tray = I3Tray()
    tray.AddModule('I3Reader','reader', FilenameList = [arg['INPUT']])
    tray.AddModule('I3MuonGun::WeightCalculatorModule', 'MuonWeight', Model=model, Generator=generator)
    #TODO find way to surpress warnings and remove failed weightings with -nan
    tray.Add("I3Writer",Filename=arg['OUTPUT'])
    tray.AddModule("TrashCan", "trash")
    tray.Execute()
    tray.Finish()

if __name__ == "__main__":
    arguments = docopt(__doc__)
    main(arguments)
