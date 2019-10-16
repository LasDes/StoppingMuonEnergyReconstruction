from icecube import icetray, dataclasses, dataio
from topologyMethods import get_muon_properties, decide_label

F = dataio.I3File("../data11058/Level3_IC86.2012_corsika.011058.000000.i3.bz2")
frame = F.pop_physics()
while frame:
	muon_bunches = get_muon_properties(frame["I3MCTree"])
	print(decide_label(muon_bunches))
	print(frame["MMCList"])
	frame = F.pop_physics()