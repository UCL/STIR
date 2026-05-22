import stir
import numpy as np
import math

stir.Verbosity.set(0)

# setting template projdata info and exam info
time_frame_def = stir.TimeFrameDefinitions()
time_frame_def.set_num_time_frames(1)
time_frame_def.set_time_frame(1, 0, 1e9)

tmpl_exam_info = stir.ExamInfo(stir.ImagingModality(stir.ImagingModality.PT))
tmpl_exam_info.patient_position = stir.PatientPosition(stir.PatientPosition.HFS)
tmpl_exam_info.set_high_energy_thres(650)
tmpl_exam_info.set_low_energy_thres(400)
tmpl_exam_info.set_time_frame_definitions(time_frame_def)

# set a radionuclide
radionuclide_db = stir.RadionuclideDB()
tmpl_exam_info.set_radionuclide(
    radionuclide_db.get_radionuclide(stir.ImagingModality(stir.ImagingModality.PT), "^18^Fluorine")
)

# define scanner geometry
detectors_per_ring = 256
number_of_rings = 48
arccorrected_bins = 180
inner_ring_radius_mm = 135.5
crystal_length_mm = 15.0
avg_interaction_depth_mm = crystal_length_mm * (1.0 - math.exp(-12.0 / crystal_length_mm))
crystal_spacing_mm = 3.313
ring_spacing_mm = crystal_spacing_mm
bin_size_mm = crystal_spacing_mm / 2
intrinsic_tilt_rad = -3.10918190  # this is the radial equivalent of 360° - 181.857°
num_axial_blocks_per_bucket = 6
num_transaxial_blocks_per_bucket = 4
num_axial_crystals_per_block = 8
num_transaxial_crystals_per_block = 8
num_crystals_per_block = 1
num_detector_layers = 1
energy_resolution = 0.14
reference_energy = 511
axial_block_spacing_mm = 27.36
transaxial_block_spacing_mm = 27.36
scanner = stir.Scanner(
    stir.Scanner.User_defined_scanner,
    "NeuroLF_15mm",
    detectors_per_ring,
    number_of_rings,
    arccorrected_bins,
    arccorrected_bins,
    inner_ring_radius_mm,
    avg_interaction_depth_mm,
    ring_spacing_mm,
    bin_size_mm,
    intrinsic_tilt_rad,
    num_axial_blocks_per_bucket,
    num_transaxial_blocks_per_bucket,
    num_axial_crystals_per_block,
    num_transaxial_crystals_per_block,
    num_crystals_per_block,
    num_crystals_per_block,
    num_detector_layers,
    energy_resolution,
    reference_energy,
    1,
    0.0,
    500.0,
    "BlocksOnCylindrical",
    crystal_spacing_mm,
    crystal_spacing_mm,
    axial_block_spacing_mm,
    transaxial_block_spacing_mm,
)

# define projection settings
span = 1
max_ring_diff = 47
num_views = 128
num_tangential_pos = arccorrected_bins
do_arc_correction = False
tmpl_proj_data_info = stir.ProjDataInfo.construct_proj_data_info(
    scanner, span, max_ring_diff, num_views, num_tangential_pos, do_arc_correction
)


def read_singles_histogram(filename, start_time=0, end_time=1e9):
    singles = np.zeros((number_of_rings, detectors_per_ring))
    singles_histogram_raw = np.fromfile(open(filename, "rb"), dtype=np.uint32)
    # the first 64 bit are the timestamp, then there are rings * detectors singles counts of 32 bit each
    for i in range(int(len(singles_histogram_raw) / (2 + number_of_rings * detectors_per_ring))):
        if i < start_time:
            continue
        if i >= end_time:
            break  # since histograms are spaced by a second, only read the ones we are interested in
        singles += singles_histogram_raw[
            i * (2 + number_of_rings * detectors_per_ring) + 2 : (i + 1)
            * (2 + number_of_rings * detectors_per_ring)
        ].reshape((number_of_rings, detectors_per_ring))
    return singles


# formula for computing lambda for singles-prompt method for randoms estimation
def lambda_formula(lambda_value, args):
    coincidence_window = args[0]
    prompts_total = args[1]
    singles_total = args[2]
    return 2 * coincidence_window * lambda_value * lambda_value - lambda_value + singles_total - prompts_total
