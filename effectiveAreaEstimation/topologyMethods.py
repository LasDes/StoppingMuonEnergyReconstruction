import numpy as np
from constantDefinitions import BARE_DET_HULL as det_hull
from constantDefinitions import DET_HULL as outer_hull
from constantDefinitions import CORE_HULL as dc_hull
from constantDefinitions import PE_THRESHOLD

def make_muon(p, prim, pe_counts):
    """There's no nice way to do this. Either it's handsome, but slow, or fast, but
    ugly. I settled for latter.

    Parameters
    ----------
    p : I3Particle
        I3 Particle Object that corresponds to a muon.

    prim : int
           The running number of the primary.

    Returns
    -------
    muon : array, shape = [23,]
           A numpy array containing everything you need to know about your favorite muon. Entries:

            ========   ======================   =======
            Element    Content                  Type
            ========   ======================   =======
            0          Zenith                   float
            1          Azimuth                  float
            2 to 4     v                        float
            5 to 7     w                        float
            8          length                   float
            9          energy                   float
            10         ptype (pdg_encoding)     int
            11         prim from above          int
            12         stop_r                   float
            13         stop_z                   float
            14         bool(stops or not)       bool
            15         bool(stops in or not)    bool
            16 to 18   entry point              float
            19 to 21   exit point               float
            22         travelled length         float
            23         minor ID                 int
            24         major ID                 int
            25         pe_count                 int
            ========   ======================   =======
    """
    muon = np.zeros(26)
    muon[0] = p.dir.zenith
    muon[1] = p.dir.azimuth
    muon[2:5] = np.array([p.dir.x, p.dir.y, p.dir.z])
    muon[5:8] = np.array([p.pos.x, p.pos.y, p.pos.z])
    muon[8] = p.length
    muon[9] = p.energy
    muon[10] = p.pdg_encoding
    muon[11] = prim
    muon[12:16] = check_stopping(muon[2:5], muon[5:8], muon[8])
    (muon[16:19], muon[19:22], muon[22]) = get_entry_exit(muon[2:5], muon[5:8],
                                                          p.length)
    muon[23] = p.id.minorID
    muon[24] = p.id.majorID
    try:
        muon[25] = pe_counts[(p.id.majorID, p.id.minorID)]
    except:
        muon[25] = 0
    return muon

def check_stopping(v, w, l):
    """Checks whether a muon is stopping strictly inside the detector.

    Parameters
    ----------
    v : array, shape = [3,]
        Direction vector of the track (must be normalized).

    w : array, shape = [3,]
        Pivot vector of the track.

    l : float
        Length of the track.

    Returns
    -------
    stop_r : float
             The r-component of the stopping point.

    stop_z : float
             the z-component of the stopping point.

    stop : bool
           Whether or not the muon stops in the detector volume.

    stop_dc : bool
              Whether or not the muon stops in deep core.
    """
    stop_point = v * l + w
    stop = outer_hull.find_simplex(stop_point) >= 0
    stop_in = det_hull.find_simplex(stop_point) >= 0
    stop_r = np.linalg.norm(stop_point[:2])
    stop_z = stop_point[2]
    return stop_r, stop_z, stop, stop_in

def get_cut(v, w, start, end):
    """A bisection algorithm that estimates an intersection between a the
    detector
    volume and a track. One day I'm going to make this prettier.

    Parameters
    ----------
    v : array, shape = [3,]
        Direction vector of the track (must be normalized).

    w : array, shape = [3,]
        Pivot vector of the track.

    start : float
            One initial value for bisection.

    end : float
          The other initial value.

    Returns
    -------
    t : float
        t value of intersection point.
    """
    t1 = start
    t2 = end
    t = (t1 + t2) / 2.0
    for k in range(20):
        if outer_hull.find_simplex(v * t + w) >= 0:
            t2 = t
        else:
            t1 = t
        t = (t1 + t2) / 2.0
    return t

def get_entry_exit(v, w, length):
    """Estimates entry and exit point of a track.

    Parameters
    ----------
    v : array, shape = [3,]
        Direction vector of the track (must be normalized).

    w : array, shape = [3,]
        Pivot vector of the track.

    length : float
             Length of the track.

    Returns
    -------
    entry : array, shape = [3,]
            Entry point.

    exit : array, shape = [3,]
           Exit point.

    travel : float
             Travelled length.
    """
    start_t = 0.0
    end_t = length
    middle_t = 0.0
    while (~(outer_hull.find_simplex(v * middle_t + w) >= 0)):
        middle_t += 10.0
        if middle_t > end_t:
            entry_t = float("nan")
            exit_t = float("nan")
            travel = 0.0
            return v * entry_t + w, v * exit_t + w, travel
    entry_t = get_cut(v, w, start_t, middle_t)
    exit_t = get_cut(v, w, end_t, middle_t)
    travel = abs(entry_t - exit_t)
    return v * entry_t + w, v * exit_t + w, travel

def get_muon_properties(mc_tree, pe_counts):
    """Builds a table of the properties of all muons in an mc_tree.

    Parameters
    ----------
    mc_tree : I3MCTree
              The MC Tree.

    Returns
    -------
    muons : array, shape = [23, N]
            A table of the properties of all N muons.
    """
    primaries = mc_tree.get_primaries()
    muons = []
    prim_number = 0
    for p in primaries:
        for d in mc_tree.get_daughters(p):
            muons += [make_muon(d, prim_number, pe_counts)]
        prim_number += 1 

    return np.array(muons)

def visited_muons(muons):
    """Does the muon actually enter the detector volume at all? This question
    is answered here.

    Parameters
    ----------
    muons : array, shape = [23, N]
            Muon table, see :meth:`make_muon` for more information on that.

    Returns
    -------
    visited : array, shape = [N,]
              Whether or not each of the muons has seen some IceCube madness.
    """
    return (muons[:, 25] >= PE_THRESHOLD)

def get_coincidence(muons):
    """Number of coincident primaries.

    Parameters
    ----------
    muons : array, shape = [23, N]
            Muon table, see :meth:`make_muon` for more information on that.
    Returns
    -------
    coincidence : int
                  The coincidence number.
    """
    return len(list(set(muons[:, 11])))

def decide_label(muons):
    """Decides whether a frame is considered a stopping event or not.

    Parameters
    ----------
    muons : array, shape = [23, N]
            Muon table, see :meth:`make_muon` for more information on that.

    Returns
    -------
    stopping : bool
               Whether it's a stopping event or not.

    stopping_dc : bool
                  Whether it's a deep core stopping event or not.
    """
    n_mu_stop = np.sum(muons[:, 15]) > 0
    all_stop = muons[visited_muons(muons), 14].all() == True
    all_stop_in = muons[visited_muons(muons), 15].all() == True
    return (all_stop & n_mu_stop), (all_stop_in & n_mu_stop)

def pe_count(mcpe_map):
    """Seeks through an I3MCPESeriesMap and counts how many PEs each particle 
    produced.

    Parameters
    ----------
    mcpe_map: I3MCPESeriesMap Object
              The Monte Carlo Pulse Series Map.

    Returns
    -------
    pre_counts: dict
                The counts in the scheme (major_id, minor_id): counts.
    """
    counts = {}
    for m in mcpe_map:
        try:
            key = (m[1][0].ID.majorID, m[1][0].ID.minorID)
            try:
                counts[key] += len(m[1])
            except:
                counts[key] = len(m[1])
        except:
            pass
    return counts