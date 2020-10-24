import os
DATA_FILE = "exportUSLab.csv"  # Points to the data Katie gave us

# Make this cross platform by using the right seperating character for the
# current operating system
SMOKE_FOLDERS = [os.path.join("data", "first_computer_full3D"),
                 os.path.join("data", "second_computer_full3D"),
                 os.path.join("data", "third_computer_full3D")]

FALSE_ALARM_FOLDERS = [os.path.join("data", "bike_full3D")]

VISUALIZE = False
# This is the NASA-stated threshold for a detector
ALARM_THRESHOLD = 0.5e-6
FALSE_ALARM_THRESHOLD = 4e-20
BIG_NUMBER = 1000
# The number of timesteps times this is what will be recored if it never alarms
NEVER_ALARMED_MULTIPLE = 1.5
# The number of timesteps times this is what will be recored if the region is infeasible
INFEASIBLE_MULTIPLE = 3
PAPER_READY = True
PLOT_TITLES = False
EPSILON = 0.000000001

# How many runs to do to evaluate the optimizer
NUM_EVALUATION_ITERS = 20

# Should plots by default be interpolated?
SMOOTH_PLOTS = False

# Change how sample points are interpolated
INTERPOLATION_METHOD = "nearest"  # "linear" "cubic"
# How many samples per axis to interpolate when visualizing
NUM_INTERPOLATION_SAMPLES=70

WORST_CASE_TTA = "worst_case_TTA"
SECOND_TTA = "second_TTA"
FASTEST_TTA = "fastest_TTA"
WORST_CASE_MC = "worst_case_MC"
SECOND_MC = "second_MC"
FASTEST_MC = "fastest_MC"

SINGLE_OBJECTIVE_FUNCTIONS_TTA = [WORST_CASE_TTA, SECOND_TTA, FASTEST_TTA]
SINGLE_OBJECTIVE_FUNCTIONS_MC = [WORST_CASE_MC, SECOND_MC, FASTEST_MC]

WORST_CASE_FUNCTIONS = [WORST_CASE_TTA, WORST_CASE_MC]
SECOND_FUNCTIONS = [SECOND_TTA, SECOND_MC]
FASTEST_FUNCTIONS = [FASTEST_TTA, FASTEST_MC]

SINGLE_OBJECTIVE_FUNCTIONS = SINGLE_OBJECTIVE_FUNCTIONS_TTA + SINGLE_OBJECTIVE_FUNCTIONS_MC
MULTI_OBJECTIVE_FUNCTIONS = [
    "multiobjective_counting", "multiobjective_competing"]
