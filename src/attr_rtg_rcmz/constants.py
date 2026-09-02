"""Frozen public constants for ATTR-RTG-RCMZ V1."""

PROTOCOL_ID = "ATTR-RTG-RCMZ-V1"
PROTOCOL_STATUS = "DRAFT V1 — LOCAL-ONLY — NOT LOCKED"
SYNTHETIC_NOTICE = "SYNTHETIC NON-OFFICIAL — NOT AN OFFICIAL WORLD OR RESULT"
CANDIDATES = ("U", "D", "L", "R", "BRAKE", "RECOVER")
TRAINING_SEEDS = (29, 43, 71, 89, 107)
ARMS = ("R", "CM", "Z", "T")
HORIZON = 8
EPISODES_PER_WORLD = 4
MAX_EPISODE_LENGTH = 64
BACKBONE_UPDATES = 2_000
SPLIT_ROWS = (
    ("train", 64, "baseline"),
    ("validation", 24, "baseline"),
    ("calibration", 24, "baseline"),
    ("test_id", 32, "baseline"),
    ("test_shift", 32, "shift"),
    ("test_ood", 32, "OOD"),
)

# Frozen statistics aliases and values.
ACTIONS = CANDIDATES
SEEDS = TRAINING_SEEDS
REGIMES = ("ID", "shift", "OOD")
TEMPERATURE_GRID = tuple(index / 4 for index in range(1, 17))
CONTRASTS = (("CM", "R"), ("CM", "Z"), ("CM", "T"), ("R", "Z"), ("R", "T"), ("Z", "T"))
BOOTSTRAP_REPLICATES = 1000
LOWER_Q = 1 / 120
UPPER_Q = 119 / 120

# Statistics aliases and fixed comparison registry.
ACTIONS = CANDIDATES
SEEDS = TRAINING_SEEDS
REGIMES = ("ID", "shift", "OOD")
TEMPERATURE_GRID = tuple(index / 4 for index in range(1, 17))
CONTRASTS = (("CM", "R"), ("CM", "Z"), ("CM", "T"), ("R", "Z"), ("R", "T"), ("Z", "T"))
BOOTSTRAP_REPLICATES = 1_000
LOWER_Q = 1 / 120
UPPER_Q = 119 / 120
