"""Numeric video contract shared by the isolated producer and model consumers.

Hand-local XY uses source pixels minus the wrist, divided by the median
wrist-to-MCP distance over valid MCPs 5, 9, 13, 17 (at least three).
The observed median must be at least eight source pixels. Screen-axis wrist
velocity uses the previous observed scale; it includes camera motion and is
not guitar-relative movement or contact evidence. Neither image position,
anatomical handedness, track identity nor GP labels are model features.
"""

SCHEMA_VERSION = 4
INPUT_REPRESENTATION = "guitar-hand-coarse-194-v1"
STRUCTURED_DIM = 194
VIEW_ORDER = ["fretting", "plucking", "unassigned_0", "unassigned_1"]
VELOCITY_SLICES = ((42, 84), (140, 182), (182, 184), (188, 190))
OBSERVATION_SLICES = ((0, 42), (84, 140), (184, 186), (186, 188), (190, 194))
MINIMUM_PALM_PIXELS = 8.
FEATURE_LAYOUT = {
    "landmarkXY": {"start": 0, "stop": 42, "points": list(range(21)), "axes": ["alongNeck", "acrossFretboard"]},
    "backwardVelocityXY": {"start": 42, "stop": 84, "units": "guitar-coordinates/second"},
    "guitarAnchorXY": {"start": 84, "stop": 96, "points": ["nut", "neckBody", "fretboardUpper", "fretboardLower", "soundhole", "bridge"]},
    "scaleGeometry": {"start": 96, "stop": 98, "names": ["neckLength/sourceDiagonal", "boardWidth/sourceDiagonal"]},
    "handLocalXY": {
        "start": 98, "stop": 140, "points": list(range(21)), "axes": ["screenX", "screenY"],
        "origin": "observedWrist", "scale": "medianValidWristToMCPSourcePixels",
        "scaleMCPs": [5, 9, 13, 17], "minimumValidMCPs": 3, "minimumScaleSourcePixels": MINIMUM_PALM_PIXELS,
    },
    "handLocalBackwardVelocityXY": {"start": 140, "stop": 182, "units": "local-palm-coordinates/second"},
    "wristBackwardVelocityXY": {
        "start": 182, "stop": 184, "axes": ["screenX", "screenY"],
        "units": "previous-observed-palm-scale/second",
        "limitation": "Includes camera motion; not guitar-relative displacement or contact.",
    },
    "palmOrientationXY": {"start": 184, "stop": 186, "axes": ["screenX", "screenY"], "direction": "unitSourcePixelWristToMiddleMCP"},
    "coarsePalmReferenceXY": {
        "start": 186, "stop": 188, "axes": ["localAlongNeck", "localAcrossBoard"],
        "origin": "initialObservedStripCenter", "scale": "fixedInitialObservedBoardWidth",
        "registration": "currentPalmMappedToPersistentInstrumentReference",
        "limitation": "Arbitrary local origin and axis, not joint=0/nut=1 or absolute low/high position.",
    },
    "coarsePalmBackwardVelocityXY": {"start": 188, "stop": 190, "units": "reference-board-widths/second"},
    "coarseCurrentNeckAxisXY": {"start": 190, "stop": 192, "axes": ["screenX", "screenY"], "direction": "registeredReferenceAlongAxisUnit"},
    "coarseBodyDistance": {
        "start": 192, "stop": 193, "units": "persistent-body-reference-scale",
        "direction": "bodywardProjectedDistanceFromPalmToSoundhole",
        "availability": "supportedExistingSoundholeAndBridgeOnly",
    },
    "coarseAxisSemanticSign": {
        "start": 193, "stop": 194, "encoding": {"bodyward": 1, "awayFromBody": -1},
        "availability": "supportedExistingBodyContextOnly; otherwise unknown and masked",
    },
}
