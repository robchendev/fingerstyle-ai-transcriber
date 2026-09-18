"""Beat-anchored, constrained symbolic timing for cleaned hypotheses."""

from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import math
from numbers import Real

import numpy as np

from .gp_events import validate_provided_timing
from .transcriber_audio import HarnessError


TICKS_PER_QUARTER = 24
GRID_STEPS = (24, 12, 8, 6, 4, 3)
SUBDIVISION_COST = {24: 0, 12: 4, 8: 16, 6: 8, 4: 24, 3: 12}
DURATION_VALUES = (
    Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1),
    Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4),
    Fraction(6), Fraction(8), Fraction(12), Fraction(16),
)
PULSE_CANDIDATES = (Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4))
MAX_LOCAL_INTERVAL_OUTLIER_RATE = .15


def _finite(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < minimum:
        raise HarnessError(f"{name} must be finite and at least {minimum}.")
    return float(value)


def _meter_duration(meter):
    if not isinstance(meter, list) or len(meter) != 2 or any(type(value) is not int or value <= 0 for value in meter):
        raise HarnessError("Meter must be a positive [numerator, denominator] pair.")
    if meter[1] & (meter[1] - 1):
        raise HarnessError("Meter denominator must be a power of two.")
    return Fraction(meter[0] * 4, meter[1])


def _tempo_quarter_bpm(tempo):
    return float(validate_provided_timing(tempo, [4, 4]))


class PerformanceMap:
    def __init__(self, evidence, metadata, duration_seconds, *, event_seconds=()):
        if not isinstance(evidence, dict) or evidence.get("kind") != "audio-beat-evidence":
            raise HarnessError("Rhythm inference requires an audio-beat-evidence document.")
        beats = np.asarray(evidence.get("beatSeconds"), dtype=np.float64)
        downbeats = np.asarray(evidence.get("downbeatSeconds"), dtype=np.float64)
        duration = _finite(duration_seconds, "Audio duration")
        if beats.ndim != 1 or not len(beats) or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0) or beats[0] < 0 or beats[-1] > duration:
            raise HarnessError("Beat evidence must contain ordered in-range beat times.")
        if downbeats.ndim != 1 or not len(downbeats) or not np.isfinite(downbeats).all() or np.any(np.diff(downbeats) <= 0):
            raise HarnessError("Beat evidence must contain ordered downbeat times.")
        indices = []
        for downbeat in downbeats:
            index = int(np.argmin(np.abs(beats - downbeat)))
            if abs(beats[index] - downbeat) > .03:
                raise HarnessError("Every downbeat must coincide with a detected beat.")
            if not indices or index > indices[-1]:
                indices.append(index)
        if not indices:
            raise HarnessError("No unique downbeat anchors remain.")
        self.metadata = metadata
        self.duration_seconds = duration
        self.beats = beats
        self.downbeat_indices = indices
        self.nominal = _NominalMap(metadata)
        self.event_seconds = tuple(sorted(_finite(value, "Retained event time") for value in event_seconds))
        self.anchor_seconds, self.anchor_quarters, self.measure_anchors = self._anchors()
        if len(self.anchor_seconds) < 2 or np.any(np.diff(self.anchor_seconds) <= 0) or np.any(np.diff(self.anchor_quarters) <= 0):
            raise HarnessError("Beat evidence did not produce a monotonic performance map.")

    def _anchors(self):
        nominal_beats = np.asarray([float(self.nominal.quarter_at(value)) for value in self.beats])
        differences = np.diff(nominal_beats)

        def pulse_cost(candidate):
            value = float(candidate)
            multiples = np.maximum(1, np.rint(differences / value))
            residual = np.abs(differences - multiples * value)
            return float(np.median(residual) + .03 * np.mean(multiples - 1))

        pulse = min(PULSE_CANDIDATES, key=lambda candidate: (pulse_cost(candidate), -float(candidate)))
        first = self.downbeat_indices[0]
        first_time = self.beats[first]
        pickup_evidence = [value for value in (*self.beats[:first], *self.event_seconds) if value < first_time - .03]
        if pickup_evidence:
            pickup_start = min(pickup_evidence)
            span = float(self.nominal.quarter_at(first_time) - self.nominal.quarter_at(pickup_start))
            first_quarter = max(0, round(span * 4) / 4)
        else:
            first_quarter = 0.0
        selected_indices = [0]
        for index in range(1, len(self.beats)):
            if nominal_beats[index] - nominal_beats[selected_indices[-1]] >= .5 * float(pulse):
                selected_indices.append(index)
        if first not in selected_indices:
            selected_indices.append(first)
            selected_indices.sort()
        sequential = {first: first_quarter}
        measure = _meter_duration(self.metadata["timeSignature"])
        for position in range(selected_indices.index(first) + 1, len(selected_indices)):
            left, right = selected_indices[position - 1], selected_indices[position]
            multiple = max(1, math.floor((nominal_beats[right] - nominal_beats[left]) / float(pulse) + .25))
            sequential[right] = sequential[left] + multiple * float(pulse)
        for position in range(selected_indices.index(first) - 1, -1, -1):
            left, right = selected_indices[position], selected_indices[position + 1]
            multiple = max(1, math.floor((nominal_beats[right] - nominal_beats[left]) / float(pulse) + .25))
            sequential[left] = sequential[right] - multiple * float(pulse)
        for downbeat in self.downbeat_indices[1:]:
            if downbeat not in sequential:
                continue
            expected = first_quarter + round((sequential[downbeat] - first_quarter) / float(measure)) * float(measure)
            if abs(sequential[downbeat] - expected) <= 1.25 * float(pulse):
                correction = expected - sequential[downbeat]
                for index in selected_indices:
                    if index >= downbeat:
                        sequential[index] += correction
        span = max(float(pulse), nominal_beats[selected_indices[-1]] - nominal_beats[selected_indices[0]])
        endpoint_drift = abs(
            (sequential[selected_indices[-1]] - sequential[selected_indices[0]])
            - (nominal_beats[selected_indices[-1]] - nominal_beats[selected_indices[0]])
        ) / span
        interval_ratios = []
        for left, right in zip(selected_indices, selected_indices[1:]):
            assigned_span = sequential[right] - sequential[left]
            if assigned_span > 0:
                interval_ratios.append((nominal_beats[right] - nominal_beats[left]) / assigned_span)
        tempo_scale = float(np.median(interval_ratios))
        local_residuals = np.abs(np.asarray(interval_ratios) / tempo_scale - 1)
        outlier_rate = float(np.mean(local_residuals > .25))
        if outlier_rate <= MAX_LOCAL_INTERVAL_OUTLIER_RATE:
            proposed = [
                (index, float(self.beats[index]), sequential[index], abs(nominal_beats[index] - sequential[index]))
                for index in selected_indices
            ]
            strategy = "sequential-beat-count"
        else:
            proposed = [(index, float(self.beats[index]), float(nominal), 0.0) for index, nominal in enumerate(nominal_beats)]
            strategy = "nominal-tempo-fallback"
        by_quarter = {}
        for value in proposed:
            current = by_quarter.get(value[2])
            if current is None or (value[3], value[1]) < (current[3], current[1]):
                by_quarter[value[2]] = value
        monotonic = sorted(by_quarter.values(), key=lambda value: value[1])
        monotonic = [value for index, value in enumerate(monotonic) if index == 0 or value[2] > monotonic[index - 1][2]]
        if len(monotonic) < 2:
            raise HarnessError("Beat evidence cannot be assigned to a monotonic metrical pulse.")
        minimum = min(value[2] for value in monotonic)
        shift = -minimum if minimum < 0 else 0
        seconds = np.asarray([value[1] for value in monotonic], dtype=np.float64)
        quarters = np.asarray([value[2] + shift for value in monotonic], dtype=np.float64)
        downbeat_set = set(self.downbeat_indices)
        measure_anchors = [{
            "timeSeconds": value[1],
            "nominalScoreQuarter": float(nominal_beats[value[0]]),
            "assignedScoreQuarter": value[2] + shift,
            "distanceToBarPhaseQuarter": min(
                (value[2] - first_quarter) % float(measure),
                float(measure) - ((value[2] - first_quarter) % float(measure)),
            ),
        } for value in monotonic if value[0] in downbeat_set]
        self.pulse_quarters = pulse
        self.bar_phase_quarters = first_quarter + shift
        self.discarded_beat_count = len(self.beats) - len(monotonic)
        self.mapping_strategy = strategy
        self.endpoint_drift_ratio = endpoint_drift
        self.tempo_scale = tempo_scale
        self.local_interval_outlier_rate = outlier_rate
        return seconds, quarters, measure_anchors

    def quarter_at(self, seconds):
        value = _finite(seconds, "Event time")
        if value < self.anchor_seconds[0]:
            slope = (self.anchor_quarters[1] - self.anchor_quarters[0]) / (self.anchor_seconds[1] - self.anchor_seconds[0])
            return Fraction(str(self.anchor_quarters[0] + (value - self.anchor_seconds[0]) * slope)).limit_denominator(1_000_000)
        if value > self.anchor_seconds[-1]:
            slope = (self.anchor_quarters[-1] - self.anchor_quarters[-2]) / (self.anchor_seconds[-1] - self.anchor_seconds[-2])
            return Fraction(str(self.anchor_quarters[-1] + (value - self.anchor_seconds[-1]) * slope)).limit_denominator(1_000_000)
        return Fraction(str(np.interp(value, self.anchor_seconds, self.anchor_quarters))).limit_denominator(1_000_000)

    def seconds_at(self, quarter):
        value = float(quarter)
        if value < self.anchor_quarters[0]:
            slope = (self.anchor_seconds[1] - self.anchor_seconds[0]) / (self.anchor_quarters[1] - self.anchor_quarters[0])
            return float(self.anchor_seconds[0] + (value - self.anchor_quarters[0]) * slope)
        if value > self.anchor_quarters[-1]:
            slope = (self.anchor_seconds[-1] - self.anchor_seconds[-2]) / (self.anchor_quarters[-1] - self.anchor_quarters[-2])
            return float(self.anchor_seconds[-1] + (value - self.anchor_quarters[-1]) * slope)
        return float(np.interp(value, self.anchor_quarters, self.anchor_seconds))

    def report(self):
        local_intervals = np.diff(self.anchor_seconds) / np.diff(self.anchor_quarters)
        return {
            "kind": "beat-anchored-performance-map",
            "anchorCount": len(self.anchor_seconds),
            "firstAnchorSeconds": float(self.anchor_seconds[0]),
            "lastAnchorSeconds": float(self.anchor_seconds[-1]),
            "firstAnchorQuarter": float(self.anchor_quarters[0]),
            "lastAnchorQuarter": float(self.anchor_quarters[-1]),
            "secondsPerQuarterMedian": float(np.median(local_intervals)),
            "secondsPerQuarterMinimum": float(np.min(local_intervals)),
            "secondsPerQuarterMaximum": float(np.max(local_intervals)),
            "pulseQuarter": [self.pulse_quarters.numerator, self.pulse_quarters.denominator],
            "barPhaseQuarter": self.bar_phase_quarters,
            "pickupEvidenceCount": int(sum(value < self.beats[self.downbeat_indices[0]] - .03 for value in self.event_seconds) + self.downbeat_indices[0]),
            "discardedBeatCount": self.discarded_beat_count,
            "mappingStrategy": self.mapping_strategy,
            "sequentialEndpointDriftRatio": float(self.endpoint_drift_ratio),
            "fittedNominalTempoScale": self.tempo_scale,
            "localIntervalOutlierRate": self.local_interval_outlier_rate,
            "measureAnchors": self.measure_anchors,
        }


class _NominalMap:
    def __init__(self, metadata):
        events = [{"timeSeconds": 0, **metadata["tempo"]}, *metadata.get("tempoChanges", [])]
        self.events = [{
                "time": time,
                "quarter": 0.0,
                "quarterBpm": _tempo_quarter_bpm(event),
                "linear": bool(event.get("linear", False)),
            } for event in events for time in [_finite(event["timeSeconds"], "Tempo event time")]]
        if any(left["time"] >= right["time"] for left, right in zip(self.events, self.events[1:])):
            raise HarnessError("Tempo events must be strictly increasing.")
        if self.events[-1]["linear"]:
            raise HarnessError("A final tempo event cannot begin an unresolved ramp.")
        quarter = 0.0
        for index in range(1, len(self.events)):
            quarter += self._segment(index - 1, self.events[index]["time"])
            self.events[index]["quarter"] = quarter

    def _segment(self, index, stop):
        event = self.events[index]
        elapsed = stop - event["time"]
        if not event["linear"]:
            return elapsed * event["quarterBpm"] / 60
        following = self.events[index + 1]
        start_bpm, end_bpm = event["quarterBpm"], following["quarterBpm"]
        if abs(end_bpm - start_bpm) < 1e-12:
            return elapsed * start_bpm / 60
        full_seconds = following["time"] - event["time"]
        quarter_width = full_seconds * (end_bpm - start_bpm) / (60 * math.log(end_bpm / start_bpm))
        slope = (end_bpm - start_bpm) / quarter_width
        return start_bpm * math.expm1(elapsed * slope / 60) / slope

    def quarter_at(self, seconds):
        value = _finite(seconds, "Nominal clock time")
        index = bisect_right([event["time"] for event in self.events], value) - 1
        return Fraction(str(self.events[index]["quarter"] + self._segment(index, value))).limit_denominator(1_000_000)


def _position_complexity(tick):
    for step in GRID_STEPS:
        if tick % step == 0:
            return SUBDIVISION_COST[step], step
    return 400, 1


def _candidate_ticks(mapper, seconds):
    raw = float(mapper.quarter_at(seconds)) * TICKS_PER_QUARTER
    result = {}
    for step in GRID_STEPS:
        base = math.floor(raw / step)
        for index in (base - 1, base, base + 1, base + 2):
            tick = max(0, index * step)
            quarter = Fraction(tick, TICKS_PER_QUARTER)
            timing_ms = abs(mapper.seconds_at(quarter) - seconds) * 1000
            complexity, actual_step = _position_complexity(tick)
            cost = round(timing_ms * 2 + complexity)
            previous = result.get(tick)
            candidate = (cost, -actual_step)
            if previous is None or candidate < previous:
                result[tick] = candidate
    return [(tick, result[tick][0]) for tick in sorted(result)]


def _optimize_onsets(document, mapper):
    try:
        from ortools.sat.python import cp_model
    except (ImportError, ModuleNotFoundError) as error:
        raise HarnessError("OR-Tools is required for constrained rhythm inference.") from error
    kinds = tuple(kind for kind in ("notes", "percussion", "techniques") if kind in document)
    times = sorted({float(event["onsetSeconds"]) for kind in kinds for event in document[kind]})
    model = cp_model.CpModel()
    variables = []
    costs = []
    regimes = {}
    triplet_selections = defaultdict(lambda: defaultdict(list))
    for index, seconds in enumerate(times):
        candidates = _candidate_ticks(mapper, seconds)
        tick = model.new_int_var_from_domain(cp_model.Domain.from_values([value[0] for value in candidates]), f"tick_{index}")
        choices = [model.new_bool_var(f"choice_{index}_{candidate_index}") for candidate_index in range(len(candidates))]
        model.add(sum(choices) == 1)
        model.add(tick == sum(value[0] * choice for value, choice in zip(candidates, choices)))
        cost = sum(value[1] * choice for value, choice in zip(candidates, choices))
        for (candidate_tick, _), choice in zip(candidates, choices):
            beat = candidate_tick // TICKS_PER_QUARTER
            position = candidate_tick % TICKS_PER_QUARTER
            if beat not in regimes:
                regimes[beat] = model.new_bool_var(f"triplet_beat_{beat}")
            regime = regimes[beat]
            if position in (4, 8, 16, 20):
                model.add(regime == 1).only_enforce_if(choice)
                triplet_selections[beat][candidate_tick].append(choice)
            elif position not in (0,):
                model.add(regime == 0).only_enforce_if(choice)
        variables.append(tick)
        costs.append(cost)
    for beat, regime in regimes.items():
        positions = triplet_selections.get(beat, {})
        if positions:
            distinct = []
            for position, choices in positions.items():
                used = model.new_bool_var(f"triplet_position_{position}")
                model.add_max_equality(used, choices)
                distinct.append(used)
            model.add(sum(distinct) >= 2 * regime)
        else:
            model.add(regime == 0)
    for left, right in zip(variables, variables[1:]):
        model.add(right >= left)
    by_time = dict(zip(times, variables))
    for string in range(1, 7):
        reattacks = sorted({float(note["onsetSeconds"]) for note in document["notes"] if note["string"] == string})
        for left, right in zip(reattacks, reattacks[1:]):
            model.add(by_time[right] > by_time[left])
    model.minimize(sum(costs) + 8 * sum(regimes.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_deterministic_time = 30
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise HarnessError("Constrained rhythm inference found no feasible onset assignment.")
    selected = {seconds: solver.value(variable) for seconds, variable in zip(times, variables)}
    result = deepcopy(document)
    timing_errors = []
    complexities = {}
    for kind in kinds:
        for event in result[kind]:
            tick = selected[float(event["onsetSeconds"])]
            quarter = Fraction(tick, TICKS_PER_QUARTER)
            event["scoreOnsetQuarter"] = [quarter.numerator, quarter.denominator]
            error = mapper.seconds_at(quarter) - float(event["onsetSeconds"])
            timing_errors.append(abs(error))
            _, step = _position_complexity(tick)
            complexities[str(step)] = complexities.get(str(step), 0) + 1
    return result, {
        "solver": "OR-Tools CP-SAT",
        "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
        "objective": solver.objective_value,
        "wallTimeSeconds": solver.wall_time,
        "eventPositionCount": len(times),
        "maximumAbsoluteTimingErrorSeconds": max(timing_errors, default=0),
        "meanAbsoluteTimingErrorSeconds": sum(timing_errors) / len(timing_errors) if timing_errors else 0,
        "eventsByFinestSubdivisionTicks": dict(sorted(complexities.items(), key=lambda item: int(item[0]), reverse=True)),
        "tripletBeatCount": sum(solver.value(value) for value in regimes.values()),
        "ticksPerQuarter": TICKS_PER_QUARTER,
        "candidateStepsTicks": list(GRID_STEPS),
        "reattackPolicy": "Distinct same-string acoustic attacks cannot collapse onto one score position.",
    }


def _optimize_durations(document):
    result = deepcopy(document)
    notes_by_string = {string: [] for string in range(1, 7)}
    for note in result["notes"]:
        notes_by_string[note["string"]].append(note)
    changes = []
    for values in notes_by_string.values():
        values.sort(key=lambda note: Fraction(*note["scoreOnsetQuarter"]))
        for index, note in enumerate(values):
            onset = Fraction(*note["scoreOnsetQuarter"])
            predicted = Fraction(str(note["notatedDurationQuarter"])).limit_denominator(1_000_000)
            next_onset = Fraction(*values[index + 1]["scoreOnsetQuarter"]) if index + 1 < len(values) else None
            candidates = list(DURATION_VALUES)
            if next_onset is not None and next_onset > onset:
                candidates.append(next_onset - onset)
                candidates = [duration for duration in candidates if onset + duration <= next_onset]
            candidates = sorted(set(duration for duration in candidates if duration > 0))
            if not candidates:
                candidates = [Fraction(1, 4)]

            def cost(duration):
                difference = abs(math.log1p(float(duration)) - math.log1p(float(predicted)))
                endpoint = onset + duration
                metric = 0 if endpoint.denominator == 1 else 1 if (endpoint * 2).denominator == 1 else 2
                long_tie = max(0, math.ceil(float(duration) / 4) - 1) * 3
                return difference * 100 + metric + long_tie

            selected = min(candidates, key=lambda duration: (cost(duration), duration))
            note["scoreDurationQuarter"] = [selected.numerator, selected.denominator]
            if selected != predicted:
                changes.append({
                    "onsetQuarter": note["scoreOnsetQuarter"],
                    "string": note["string"],
                    "predictedQuarter": [predicted.numerator, predicted.denominator],
                    "selectedQuarter": note["scoreDurationQuarter"],
                })
    return result, {
        "policy": "Predicted notated duration is a soft log-distance prior over conventional values, bounded by the next same-string attack.",
        "changedCount": len(changes),
        "changes": changes,
    }


def infer_notated_timing(document, evidence, *, include_unsupported_tail=False):
    if not isinstance(document, dict) or not isinstance(document.get("metadata"), dict):
        raise HarnessError("Rhythm inference requires hypothesis metadata.")
    mapper = PerformanceMap(
        evidence,
        document["metadata"],
        document["audioDurationSeconds"],
        event_seconds=[event["onsetSeconds"] for kind in ("notes", "percussion") for event in document[kind]],
    )
    if type(include_unsupported_tail) is not bool:
        raise TypeError("include_unsupported_tail must be boolean.")
    beat_supported_end = float(mapper.anchor_seconds[-1])
    bounded = deepcopy(document)
    unsupported = {}
    for kind in tuple(kind for kind in ("notes", "percussion", "techniques") if kind in document):
        unsupported[kind] = sum(event["onsetSeconds"] > beat_supported_end for event in document[kind])
        if not include_unsupported_tail:
            bounded[kind] = [event for event in document[kind] if event["onsetSeconds"] <= beat_supported_end]
    if not bounded["notes"] and not bounded["percussion"]:
        raise HarnessError("No events remain inside beat-supported timing.")
    onsets, onset_report = _optimize_onsets(bounded, mapper)
    durations, duration_report = _optimize_durations(onsets)
    score_end_seconds = document["audioDurationSeconds"] if include_unsupported_tail else beat_supported_end
    raw_end = mapper.quarter_at(score_end_seconds)
    end = Fraction(math.ceil(float(raw_end) * TICKS_PER_QUARTER), TICKS_PER_QUARTER)
    durations["scoreAudioEndQuarter"] = [end.numerator, end.denominator]
    durations["scoreTimeAnchors"] = [
        {"scoreQuarter": float(quarter), "audioSeconds": float(seconds)}
        for quarter, seconds in zip(mapper.anchor_quarters, mapper.anchor_seconds)
    ]
    durations["notatedTimeSignatureChanges"] = [{
        "scoreQuarter": list(_round_score_position(mapper.quarter_at(event["timeSeconds"]))),
        "timeSignature": event["timeSignature"],
    } for event in document["metadata"].get("timeSignatureChanges", [])]
    durations["notatedTempoChanges"] = [{
        "scoreQuarter": list(_round_score_position(mapper.quarter_at(event["timeSeconds"]))),
        "bpm": event["bpm"],
        "beatUnit": event["beatUnit"],
        "linear": event.get("linear", False),
    } for event in document["metadata"].get("tempoChanges", [])]
    meter = _meter_duration(document["metadata"]["timeSignature"])
    pickup = Fraction(str(mapper.bar_phase_quarters)) % meter
    durations["pickupDurationQuarter"] = [pickup.numerator, pickup.denominator]
    initial = document["metadata"]["tempo"]
    if mapper.mapping_strategy == "sequential-beat-count":
        unit = Fraction(*initial["beatUnit"])
        quarter_bpm = 60 / mapper.report()["secondsPerQuarterMedian"]
        durations["notatedInitialTempo"] = {
            "bpm": quarter_bpm / float(unit * 4),
            "beatUnit": initial["beatUnit"],
            "linear": False,
        }
    return durations, {
        "schemaVersion": 1,
        "kind": "notation-rhythm-inference",
        "performanceMap": mapper.report(),
        "onsetOptimization": onset_report,
        "durationOptimization": duration_report,
        "beatSupportedEndSeconds": beat_supported_end,
        "eventsAfterBeatSupport": unsupported,
        "beatUnsupportedTailIncluded": include_unsupported_tail,
        "rawHypothesesModified": False,
    }


def _round_score_position(value):
    result = Fraction(round(float(value) * TICKS_PER_QUARTER), TICKS_PER_QUARTER)
    return result.numerator, result.denominator


def score_seconds_at(document, quarter):
    anchors = document.get("scoreTimeAnchors")
    if not isinstance(anchors, list) or len(anchors) < 2:
        raise HarnessError("Missing score/audio anchors for a newly inferred score position; run beat-based rhythm inference.")
    if any(not isinstance(point, dict) or not {"scoreQuarter", "audioSeconds"} <= point.keys() for point in anchors):
        raise HarnessError("Score/audio anchors require scoreQuarter and audioSeconds.")
    quarters = [_finite(point["scoreQuarter"], "Score anchor") for point in anchors]
    seconds = [_finite(point["audioSeconds"], "Audio anchor") for point in anchors]
    if any(a >= b for a, b in zip(quarters, quarters[1:])) or any(a >= b for a, b in zip(seconds, seconds[1:])):
        raise HarnessError("Score/audio anchors must be strictly increasing.")
    value = float(quarter)
    if value < quarters[0]:
        time = seconds[0] + (value - quarters[0]) * (seconds[1] - seconds[0]) / (quarters[1] - quarters[0])
    elif value > quarters[-1]:
        time = seconds[-1] + (value - quarters[-1]) * (seconds[-1] - seconds[-2]) / (quarters[-1] - quarters[-2])
    else:
        time = float(np.interp(value, quarters, seconds))
    if time < 0:
        raise HarnessError("Inferred score position precedes the supplied audio.")
    return time
