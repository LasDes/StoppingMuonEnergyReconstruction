from math import sqrt
import os
import glob

import numpy as np
import scipy.spatial as spatial

from icecube import icetray, dataclasses, dataio, simclasses, gulliver
from icecube import linefit, paraboloid, millipede, common_variables, cramer_rao
from I3Tray import *
from icecube.weighting import CORSIKAWeightCalculator
from icecube.weighting import fluxes

from classDefinitions import *
import constantDefinitions as CONST
from globalMethods import *
from topologyMethods import get_muon_properties, decide_label, get_coincidence, pe_count, visited_muons

# # # # # # # # # #
# Icetray Modules #
# # # # # # # # # #

#==============================================================================
# PreCuts
#==============================================================================
class PreCuts(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("Cuts", "Cuts to Apply to data.",
                          [87.0, 0.5, 250.0])
        self.AddParameter("Geometry", "Geometry Object.")

    def Configure(self):
        print("Precuts starting...")
        self.cuts = self.GetParameter("Cuts")
        self.shell = VetoRegion("v1", False)
        self.shell.make_shell(CONST.SHELL_0_OMS, CONST.SHELL_0_STRINGS, True)
        self.geometry_data = self.GetParameter("Geometry")

    """Physics Frame Method. Applies Precuts as configured.

    Parameters
    ----------
    frame : I3Frame Object
            A Physics frame.
    """
    def Physics(self, frame):
        # Calculate all necessary features to apply cuts
        if frame.Has("BelowCuts"):
            self.PushFrame(frame)
        else:
            fit = frame["SplineMPE"]
            zenith = fit.dir.zenith
            pulses = frame["HVInIcePulses"].apply(frame)
            pulselist = []
            for omkey, pulse_series in pulses:
                for pulse in pulse_series:
                    pos = self.geometry_data.get_pos(omkey.om, omkey.string)
                    pulselist += [[pulse.charge, pulse.time, 
                                  self.shell.check_mask(omkey.om, omkey.string),
                                  pos[0], pos[1]
                                  ]]
            pulselist = np.array(pulselist)

            qratio =  np.sum(pulselist[pulselist[:, 2] == 1, 0]) / np.sum(pulselist[:, 0])
            srt = np.argsort(pulselist[:, 1])[-4:]
            stop_x = np.sum(pulselist[srt, 3] * pulselist[srt, 0]) / np.sum(pulselist[srt, 0])
            stop_y = np.sum(pulselist[srt, 4] * pulselist[srt, 0]) / np.sum(pulselist[srt, 0])
            stop_r = sqrt(stop_x**2 + stop_y**2)

            # Apply Cuts
            if zenith > self.cuts[0] / 180 * np.pi or qratio > self.cuts[1] or stop_r > self.cuts[2]:
                frame["BelowCuts"] = dataclasses.I3Double(0.0)
            else:
                frame["BelowCuts"] = dataclasses.I3Double(1.0)
            self.PushFrame(frame)

    def DAQ(self, frame):
        self.PushFrame(frame)


#==============================================================================
# GetLabels
#==============================================================================
class GetLabels(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "Features",
            "Object from class ExtractedFeatures to contain extracted features.",
            ExtractedFeatures())

    def add(self, key, value, role="label"):
        self.feature.add(key, value, role)

    def Configure(self):
        print("GetLabels Starting...")
        self.feature = self.GetParameter("Features")
        self.frame_index = 0
        self.file_index = 0

    def Physics(self, frame):
        mep_t = dataclasses.get_most_energetic_primary(frame[
            "I3MCTree"])
        traj = Trajectory(
            mep_t.dir.x, mep_t.dir.y,
            mep_t.dir.z, mep_t.pos.x,
            mep_t.pos.y, mep_t.pos.z)
        self.add("zenith_true", mep_t.dir.zenith)
        self.add("azimuth_true", mep_t.dir.azimuth)
        self.add("energy_mep", mep_t.energy)

        try:
            pe_counts = pe_count(frame["I3MCPESeriesMap"])
        except:
            pe_counts = 0
        muon_bunches = get_muon_properties(frame["I3MCTree"], pe_counts)

        self.add("coincidence", get_coincidence(muon_bunches))
        visited = visited_muons(muon_bunches)
        stopping = muon_bunches[:, 15] == True
        energy_total = np.mean(muon_bunches[:, 9])

        if np.sum(visited) > 0:
            stopr = np.mean(muon_bunches[visited, 12])
            stopz = np.mean(muon_bunches[visited, 13])
            nmust = np.sum(muon_bunches[visited, 15])
            energy_stop = np.mean(muon_bunches[stopping, 9])
            stop_det, stop_dc = decide_label(muon_bunches)
        else:
            stopr = NaN
            stopz = NaN
            nmust = 0
            energy_stop = NaN
            stop_det = False
            stop_dc = False
        self.add("true_stop_r", stopr)
        self.add("true_stop_z", stopz)
        self.add("energy_stop", energy_stop)
        self.add("energy_total", energy_total)
        self.add("n_mu", len(muon_bunches))
        self.add("n_mu_stop", nmust)
        self.add("label_det", stop_det)
        self.add("label_in", stop_dc)
        self.add("frame_index", self.frame_index)
        self.frame_index += 1
        self.PushFrame(frame)

    def DAQ(self, frame):
        self.PushFrame(frame)

    def Finish(self):
        print("Finished GetLabels.")


#==============================================================================
# GetAttributes
#==============================================================================
class GetAttributes(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter(
            "Features",
            "Object from class ExtractedFeatures to contain extracted features.",
            ExtractedFeatures())
        self.AddParameter("PulseKeyList", "Keys to pulses to be used.")
        self.AddParameter("RegionList", "List of Veto Regions")
        self.AddParameter("Geometry", "Geometry Object.")
        self.AddParameter("FitKey", "Fit to be used.")
        self.AddParameter("ParameterKey", "Fit to be used.")
        self.AddParameter("Pulse", "Whether to calculate Pulse Attributes.",
                          True)
        self.AddParameter("CV", "Whether to calculate CV Attributes.", True)
        self.AddParameter("Other", "Whether to calculate other Attributes.",
                          True)

    def add(self, key, name, role="attribute"):
        self.feature.add(key, name, role)

    def Configure(self):
        print("GetAttributes Starting...")
        self.feature = self.GetParameter("Features")
        self.pulse_key_list = self.GetParameter("PulseKeyList")
        self.fit_key = self.GetParameter("FitKey")
        self.parameter_key = self.GetParameter("ParameterKey")
        self.geometry_data = self.GetParameter("Geometry")
        self.pulse = self.GetParameter("Pulse")
        self.cv = self.GetParameter("CV")
        self.other = self.GetParameter("Other")
        self.first_pass = False
        # All DOMs
        veto0 = VetoRegion("v0", True)

        # Outermost Shell
        veto1 = VetoRegion("v1", False)
        veto1.make_shell(CONST.SHELL_0_OMS, CONST.SHELL_0_STRINGS, True)

        # Second-to-outermost Shell
        veto2 = VetoRegion("v2", False)
        veto2.make_shell(CONST.SHELL_1_OMS, CONST.SHELL_1_STRINGS, True)

        # Third-to-outermost Shell
        veto3 = VetoRegion("v3", False)
        veto3.make_shell(CONST.SHELL_2_OMS, CONST.SHELL_2_STRINGS, True)

        # Fourth-to-outermost Shell
        veto4 = VetoRegion("v4", False)
        veto4.make_shell(CONST.SHELL_3_OMS, CONST.SHELL_3_STRINGS, True)

        # The core (not "Deep Core", just core, w/e)
        veto5 = VetoRegion("v5", True)
        veto5.make_shell(CONST.SHELL_3_OMS, CONST.SHELL_3_STRINGS, False)
        self.region_list = [veto0, veto1, veto2, veto3, veto4, veto5]

    def Physics(self, frame):
        traj = self.fitAttributes(frame[self.fit_key],
                                         frame[self.parameter_key],
                                         frame["SplineMPECramerRaoParams"])
        if self.pulse == True:
            self.pulseAttributes(frame, traj)
        if self.cv == True:
            for key in self.pulse_key_list:
                self.commonVariables(frame[key].apply(frame),
                                     frame[self.fit_key], key)
        if self.other == True:
            self.miscAttributes(frame)
        self.first_pass = True
        self.PushFrame(frame)

    """Generate attributes that rely on the results of a fit.
    
    Parameters
    ----------
    fit : I3Particle Object
          The Fit to be used.

    param : I3LineFitParams Object
            Line Fit Parameters.

    cr : I3CramerRaoParams Object
         Cramer Rao Parameters.
    """
    def fitAttributes(self, fit, param, cr):
        traj = Trajectory(fit.dir.x, fit.dir.y, fit.dir.z,
                                 fit.pos.x, fit.pos.y, fit.pos.z,
                                 fit.dir.zenith)

        self.add("zenith_%s" % (self.fit_key), fit.dir.zenith)
        self.add("azimuth_%s" % (self.fit_key), fit.dir.azimuth)
        self.add("dirx_%s" % (self.fit_key), fit.dir.x)
        self.add("diry_%s" % (self.fit_key), fit.dir.y)
        self.add("dirz_%s" % (self.fit_key), fit.dir.z)
        self.add("posx_%s" % (self.fit_key), fit.pos.x)
        self.add("posy_%s" % (self.fit_key), fit.pos.y)
        self.add("posz_%s" % (self.fit_key), fit.pos.z)
        self.add("closestappr_%s" % (self.fit_key),
                 traj.closest_approach())
        self.entry, self.exit = traj.travel_length()
        self.add("exit_z_%s" % (self.fit_key),
                 traj.eval(self.exit)[1])
        self.add("travellength_%s" % (self.fit_key),
                 abs(self.entry - self.exit))
        self.add("CR_theta_%s" % (self.fit_key), cr.cramer_rao_theta)
        self.add("CR_phi_%s" % (self.fit_key), cr.cramer_rao_phi)
        self.add("CR_vartheta_%s" % (self.fit_key), cr.variance_theta)
        self.add("CR_varphi_%s" % (self.fit_key), cr.variance_phi)
        self.add("CR_varx_%s" % (self.fit_key), cr.variance_x)
        self.add("CR_vary_%s" % (self.fit_key), cr.variance_y)
        self.add("CR_covthetaphi_%s" % (self.fit_key),
                 cr.covariance_theta_phi)
        self.add("CR_covthetax_%s" % (self.fit_key), cr.covariance_theta_x)
        self.add("CR_covthetay_%s" % (self.fit_key), cr.covariance_theta_y)
        self.add("CR_covphix_%s" % (self.fit_key), cr.covariance_phi_x)
        self.add("CR_covphiy_%s" % (self.fit_key), cr.covariance_phi_y)
        self.add("CR_covxy_%s" % (self.fit_key), cr.covariance_x_y)
        dust_B, dust_A = dust_coeff(traj.v, traj.w,
                                    self.entry, self.exit)
        self.add("dustiness_b_%s" % (self.fit_key), dust_B)
        self.add("dustiness_a_%s" % (self.fit_key), dust_A)

        if self.parameter_key != None:
            if type(param) == linefit.icecube.recclasses.I3LineFitParams:
                self.add("lfvel_%s" % (self.fit_key), param.LFVel)
                self.add("lfvel_x_%s" % (self.fit_key), param.LFVelX)
                self.add("lfvel_y_%s" % (self.fit_key), param.LFVelY)
                self.add("lfvel_z_%s" % (self.fit_key), param.LFVelZ)
                self.add("NHits_%s" % (self.fit_key), param.NHits)
            else:
                self.add("logl_%s" % (self.fit_key), param.logl)
                self.add("ndof_%s" % (self.fit_key), param.ndof)
                self.add("nmini_%s" % (self.fit_key), param.nmini)
                self.add("rlogl_%s" % (self.fit_key), param.rlogl)
        return traj

    """Gather all Attributes corresponding to a pulsemap.
    
    Parameters
    ----------
    curr_frame : I3Frame Object
                 Current frame.

    traj : Trajectory Object
           Trajectory of the reconstruction.
    """
    def pulseAttributes(self, curr_frame, traj):
        # Walk through all pulse keys
        for key in self.pulse_key_list:
            pulses = curr_frame[key].apply(curr_frame)
            length = 0
            # Get length of pulse list first to speed up execution.
            for omkey, pulse_series in pulses:
                length += len(pulse_series)
            pulse_table = np.zeros((length, 11 + len(self.region_list)))
            _i = 0

            # Walk through all pulses and gather attribtues such as charge,
            # times, etc.
            for omkey, pulse_series in pulses:
                for pulse in pulse_series:
                    pulse_table[_i, 0] = omkey.om
                    pulse_table[_i, 1] = omkey.string
                    for _j in range(len(self.region_list)):
                        pulse_table[_i, 11 + _j] = self.region_list[
                            _j].check_mask(omkey.om, omkey.string)

                    pulse_table[_i, 2:5] = self.geometry_data.get_pos(
                        pulse_table[_i, 0], pulse_table[_i, 1])
                    pulse_table[_i, 5] = pulse.time
                    pulse_table[_i, 6] = pulse.charge
                    pulse_table[_i, 7] = traj.project_onto(pulse_table[
                        _i, 2:5])
                    pulse_table[_i, 8] = traj.get_distance(pulse_table[
                        _i, 2:5])
                    _i += 1

            pulse_table[:, 9] = (pulse_table[:, 8] < 300.0) & (
                pulse_table[:, 7] < (self.exit - self.entry) / 2.0)
            pulse_table[:, 10] = (pulse_table[:, 8] < 300.0) & (
                pulse_table[:, 7] >= (self.exit - self.entry) / 2.0)

            # Calculate all charge sums if wanted.
            for _j in range(len(self.region_list)):
                self.add(
                    "charge_%s_%s_corridor_out" %
                    (self.region_list[_j].name, key),
                    np.sum(pulse_table[(pulse_table[:, 11 + _j] == True) & (
                        pulse_table[:, 10] == True), 6]))
                self.add(
                    "charge_%s_%s_corridor_in" %
                    (self.region_list[_j].name, key),
                    np.sum(pulse_table[(pulse_table[:, 11 + _j] == True) & (
                        pulse_table[:, 9] == True), 6]))
                self.add("charge_%s_%s_all_out" % (
                    self.region_list[_j].name, key
                ), np.sum(pulse_table[(pulse_table[:, 11 + _j] == True), 6]))
                self.add("charge_%s_%s_all_in" %
                         (self.region_list[_j].name, key),
                         np.sum(pulse_table[(pulse_table[:, 11 + _j] == False
                                              ), 6]))

            # Calculate the volume and area of the convex hull and associated
            # features.
            try:
                hull = spatial.ConvexHull(pulse_table[:, 2:5])
            except:
                self.add("convhull_vol_%s" % (key), 0.0)
                self.add("convhull_surf_%s" % (key), 0.01)
            else:
                self.add("convhull_vol_%s" % (key), hull.volume)
                self.add("convhull_surf_%s" % (key), hull.area)

            self.add("sphericalness_%s" % (key), hull.volume
                     **0.6667 / hull.area)
            charge_sum = np.sum(pulse_table[:, 6])
            self.add("Qdensity_%s" % (key), charge_sum / hull.volume)

            # Calculate center of gravity and associated features
            cog = np.array(
                [np.sum(pulse_table[:, 2] * pulse_table[:, 6]) / charge_sum,
                 np.sum(pulse_table[:, 3] * pulse_table[:, 6]) / charge_sum,
                 np.sum(pulse_table[:, 4] * pulse_table[:, 6]) / charge_sum])
            self.add("charge_cog_r_%s" % (key), sqrt(cog[0]**2 + cog[1]**2))
            self.add("charge_cog_z_%s" % (key), cog[2])

            # Calculate statistic quantities of pulse times and locations.
            proj_mean, proj_std = weighted_mean_std(pulse_table[:, 7],
                                                     pulse_table[:, 6])
            dist_mean, dist_std = weighted_mean_std(pulse_table[:, 8],
                                                     pulse_table[:, 6])
            time_mean, time_std = weighted_mean_std(pulse_table[:, 5],
                                                     pulse_table[:, 5])
            self.add(
                "proj_frame_%s" % (key),
                abs(np.max(pulse_table[:, 7]) - np.min(pulse_table[:, 7])))
            self.add("proj_std_%s" % (key), proj_std)
            self.add("dist_mean_%s" % (key), dist_mean)
            self.add("dist_std_%s" % (key), dist_std)
            self.add(
                "time_frame_%s" % (key),
                abs(np.max(pulse_table[:, 5]) - np.min(pulse_table[:, 5])))
            self.add("time_std_%s" % (key), time_std)
            latest_pulses = pulse_table[pulse_table[:, 5].argsort()]
            for k in range(1, 10):
                try:
                    latest_k_pulses = latest_pulses[-k:, :]
                    stop_point = np.array([
                        np.sum(latest_k_pulses[:, 2] * latest_k_pulses[:, 6]),
                        np.sum(latest_k_pulses[:, 3] * latest_k_pulses[:, 6]),
                        np.sum(latest_k_pulses[:, 4] * latest_k_pulses[:, 6])
                    ])
                    stop_point /= np.sum(latest_k_pulses[:, 6])

                    self.add("stop_point_r_%i_%s" % (k, key),
                             sqrt(stop_point[0]**2 + stop_point[1]**2))
                    self.add("stop_point_z_%i_%s" % (k, key), stop_point[2])
                except:
                    self.add("stop_point_r_%i_%s" % (k, key), np.nan)
                    self.add("stop_point_z_%i_%s" % (k, key), np.nan)

                stop_point_proj = traj.project_onto(stop_point)
                self.add("stop_point_proj_%i_%s" % (k, key), stop_point_proj)
                self.add("proj_track_%i_%s" % (k, key),
                         abs(stop_point_proj - self.entry))
            self.add(
                "freelength_%s" % (key), abs(self.exit - self.entry) -
                abs(np.max(pulse_table[:, 7]) - np.min(pulse_table[:, 7])))

            # Calculate empty projected bins or whatever you call it.
            bin_widths = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
            for bw in bin_widths:
                hist = fixed_width_hist(pulse_table[:, 7], self.entry,
                                        self.exit, bw,
                                        weights=pulse_table[:, 6])

                self.add("emptybins_%i_%s" % (int(bw), key),
                         np.sum(hist < 1.0) / float(len(hist)))
                self.add("emptybins_%i_N_%s" % (int(bw), key),
                         np.sum(hist == 0))
                self.add("bins_frame_%i_%s" % (int(bw), key),
                         np.max(hist) - np.min(hist))
                self.add("bins_std_%i_%s" % (int(bw), key),
                         np.std(hist))
                self.add("bins_mean_%i_%s" % (int(bw), key),
                         np.mean(hist))

                hist_bs = fixed_width_hist(pulse_table[:, 7], self.entry,
                                           stop_point_proj, bw,
                                           weights=pulse_table[:, 6])
                self.add("emptybins_%i_BS_%s" % (int(bw), key),
                         np.sum(hist_bs < 1.0) / float(len(hist_bs)))
                self.add("bins_frame_%i_BS_%s" % (int(bw), key),
                         np.max(hist_bs) - np.min(hist_bs))
                self.add("bins_std_%i_BS_%s" % (int(bw), key),
                         np.std(hist_bs))
                self.add("bins_mean_%i_BS_%s" % (int(bw), key),
                         np.mean(hist_bs))

    """Gather all Attributes from the common_variables module
    
    Parameters
    ----------
    pulses : I3PulseSeriesMap Object
             Pulse map

    traj : Trajectory Object
           Trajectory corresponding to the reconstruction

    key : str
          Key of the pulse map used.
    """
    def commonVariables(self, pulses, traj, key):
        trackchar = common_variables.track_characteristics.calculate_track_characteristics(
            self.geometry_data.G, pulses, traj, 300.0)
        directhits = common_variables.direct_hits.calculate_direct_hits(
            self.geometry_data.G, pulses, traj)["A"]
        hitmult = common_variables.hit_multiplicity.calculate_hit_multiplicity(
            self.geometry_data.G, pulses)
        hitstat = common_variables.hit_statistics.calculate_hit_statistics(
            self.geometry_data.G, pulses)
        self.add("cv_addqtd_%s" % key, trackchar.avg_dom_dist_q_tot_dom)
        self.add("cv_ehtl_%s" % key, trackchar.empty_hits_track_length)
        self.add("cv_thsl_%s" % key, trackchar.track_hits_separation_length)
        self.add("cv_thds_%s" % key,
                 trackchar.track_hits_distribution_smoothness)

        self.add("cv_n_dir_strings_%s" % key, directhits.n_dir_strings)
        self.add("cv_n_dir_doms_%s" % key, directhits.n_dir_doms)
        self.add("cv_n_dir_pulses_%s" % key, directhits.n_dir_pulses)
        self.add("cv_n_early_strings_%s" % key, directhits.n_early_strings)
        self.add("cv_n_early_doms_%s" % key, directhits.n_early_doms)
        self.add("cv_n_early_pulses_%s" % key, directhits.n_early_pulses)
        self.add("cv_q_early_pulses_%s" % key, directhits.q_early_pulses)
        self.add("cv_n_late_strings_%s" % key, directhits.n_late_strings)
        self.add("cv_n_late_doms_%s" % key, directhits.n_late_doms)
        self.add("cv_n_late_pulses_%s" % key, directhits.n_late_pulses)
        self.add("cv_q_late_pulses_%s" % key, directhits.q_late_pulses)
        self.add("dir_track_length_%s" % key, directhits.dir_track_length)
        self.add("dh_smoothness_%s" % key,
                 directhits.dir_track_hit_distribution_smoothness)

        self.add("n_hit_strings_%s" % key, hitmult.n_hit_strings)
        self.add("n_hit_doms_%s" % key, hitmult.n_hit_doms)
        self.add("n_hit_doms_1_%s" % key, hitmult.n_hit_doms_one_pulse)
        self.add("n_pulses_%s" % key, hitmult.n_pulses)

        self.add("cog_%s" % key, hitstat.cog[2])
        self.add("cog_z_%s" % key, hitstat.cog_z_sigma)
        self.add("min_pulse_time_%s" % key, hitstat.min_pulse_time)
        self.add("max_pulse_time_%s" % key, hitstat.max_pulse_time)
        self.add("q_max_doms_%s" % key, hitstat.q_max_doms)
        self.add("q_tot_pulses_%s" % key, hitstat.q_tot_pulses)
        self.add("z_min_%s" % key, hitstat.z_min)
        self.add("z_max_%s" % key, hitstat.z_max)
        self.add("z_mean_%s" % key, hitstat.z_mean)
        self.add("z_sigma_%s" % key, hitstat.z_sigma)
        self.add("z_travel_%s" % key, hitstat.z_travel)

    """Gather misc attributes.

    Parameters
    ----------
    curr_frame : I3Frame Object
                 Current frame.
    """
    def miscAttributes(self, curr_frame):
        if curr_frame.Has("SplineMPETruncatedEnergy_SPICEMie_ORIG_dEdX"):
            self.add(
                "de_dx",
                curr_frame["SplineMPETruncatedEnergy_SPICEMie_ORIG_dEdX"].value)
        else:
            self.add("de_dx", 0.0)
        if curr_frame.Has("SplineMPEMuEXDifferential"):
            self.add("energy_muex", curr_frame["SplineMPEMuEXDifferential"].energy)
        else:
            self.add("energy_muex", 0.0)
        if curr_frame.Has("SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Muon"):
            self.add("energy_trun_AD_mu", curr_frame[
                "SplineMPETruncatedEnergy_SPICEMie_AllDOMS_Muon"].energy)
        else:
            self.add("energy_trun_AD_mu", 0.0)
        if curr_frame.Has("SplineMPETruncatedEnergy_SPICEMie_DOMS_Muon"):
            self.add(
                "energy_trun_D_mu",
                curr_frame["SplineMPETruncatedEnergy_SPICEMie_DOMS_Muon"].energy)
        else:
            self.add("energy_trun_D_mu", 0.0)
        if curr_frame.Has("SplineMPETruncatedEnergy_SPICEMie_ORIG_Muon"):
            self.add(
                "energy_trun_O_mu",
                curr_frame["SplineMPETruncatedEnergy_SPICEMie_ORIG_Muon"].energy)
        else:
            self.add("energy_trun_O_mu", 0.0)

    def DAQ(self, frame):
        self.PushFrame(frame)

    def Finish(self):
        print("GetAttributes Finished.")

#==============================================================================
# AddFeaturesToI3
#==============================================================================
class AddFeaturesToI3(icetray.I3ConditionalModule):
    def __init__(self, context):
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("Features", "Feature Class Object.")

    def Configure(self):
        self.feature = self.GetParameter("Features")

    def Physics(self, frame):
        labels, attributes = self.feature.get_row()
        frame["Hoinka_Labels"] = dataclasses.I3MapStringDouble(labels)
        for key, value in attributes.iteritems():
            prefixed_key = "Hoinka_" + key
            frame[prefixed_key] = dataclasses.I3MapStringDouble({prefixed_key:
                                                                 value})
        self.PushFrame(frame)

    def DAQ(self, frame):
        self.PushFrame(frame)

    def Finish(self):
        print("AddFeaturesToI3 Finished.")
