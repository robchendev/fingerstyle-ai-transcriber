"""Appended schema-v5 fretboard feature contract."""

from .video_features import STRUCTURED_DIM as BASE_STRUCTURED_DIM


SCHEMA_VERSION = 5
INPUT_REPRESENTATION = "guitar-hand-fretboard-233-v1"
STRUCTURED_DIM = 233
FINGERTIP_LANDMARKS = (4, 8, 12, 16, 20)
APPENDED_FEATURE_LAYOUT = {
    "fingertipScaleString": {
        "start": 194, "stop": 204, "points": list(FINGERTIP_LANDMARKS),
        "axes": ["nut0Bridge1", "string6ZeroString1Five"],
    },
    "fingertipFret": {"start": 204, "stop": 209, "points": list(FINGERTIP_LANDMARKS)},
    "fingertipNearestStringDistance": {
        "start": 209, "stop": 214, "points": list(FINGERTIP_LANDMARKS),
        "units": "inter-string spacings",
    },
    "fingertipInsideStringSpan": {"start": 214, "stop": 219, "points": list(FINGERTIP_LANDMARKS)},
    "fingertipScaleStringVelocity": {
        "start": 219, "stop": 229, "points": list(FINGERTIP_LANDMARKS),
        "axes": ["scaleLengthsPerSecond", "stringSpacingsPerSecond"],
    },
    "geometryQuality": {
        "start": 229, "stop": 233,
        "names": ["confidence", "ageSeconds", "flowErrorBoardWidths", "detectorAnchor"],
    },
}
FEATURE_LAYOUT = {
    "base": {"start": 0, "stop": BASE_STRUCTURED_DIM, "schemaVersion": 4},
    **APPENDED_FEATURE_LAYOUT,
}


def validate_contract():
    indices = []
    for value in APPENDED_FEATURE_LAYOUT.values():
        indices.extend(range(value["start"], value["stop"]))
    if BASE_STRUCTURED_DIM != 194 or sorted(indices) != list(range(BASE_STRUCTURED_DIM, STRUCTURED_DIM)):
        raise RuntimeError("Schema-v5 appended features must cover dimensions 194 through 232 exactly.")


validate_contract()
