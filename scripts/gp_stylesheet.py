"""Clear inherited template attribution while preserving notation styles."""

from .transcriber_audio import HarnessError


ATTRIBUTION_FIELDS = ("Artist", "Album", "Words", "Music", "WordsAndMusic", "Tabber")


def _binary_stylesheet(data):
    if len(data) < 4:
        raise HarnessError("Truncated binary GP stylesheet.")
    count = int.from_bytes(data[:4], "big")
    position = 4
    output = bytearray(data[:4])
    changed = []

    def take(length):
        nonlocal position
        if length < 0 or position + length > len(data):
            raise HarnessError("Truncated binary GP stylesheet value.")
        value = data[position:position + length]
        position += length
        return value

    if count > (len(data) - 4) // 3:
        raise HarnessError("Invalid binary GP stylesheet entry count.")
    for _ in range(count):
        start = position
        key = take(take(1)[0]).decode("utf-8")
        kind = take(1)[0]
        value_start = position
        if kind == 3:
            value = take(int.from_bytes(take(2), "big"))
            if key in {f"Header/{field}" for field in ATTRIBUTION_FIELDS} and value:
                output.extend(data[start:value_start] + b"\x00\x00")
                changed.append(key)
                continue
        elif kind in {0, 1, 2, 4, 5, 6, 7}:
            take({0: 1, 1: 4, 2: 4, 4: 8, 5: 8, 6: 16, 7: 4}[kind])
        else:
            raise HarnessError(f"Unsupported GP stylesheet value type: {kind}")
        output.extend(data[start:position])
    if position != len(data):
        raise HarnessError("Unexpected trailing binary GP stylesheet data.")
    return bytes(output), changed


def _varint(data, position):
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise HarnessError("Truncated GP stylesheet varint.")
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, position
    raise HarnessError("Oversized GP stylesheet varint.")


def _encode_varint(value):
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def _clear_text_fields(data, paths):
    position = 0
    output = bytearray()
    changed = 0
    while position < len(data):
        start = position
        tag, position = _varint(data, position)
        field, wire = tag >> 3, tag & 7
        if field == 0:
            raise HarnessError("Invalid GP stylesheet field number.")
        tag_end = position
        if wire == 0:
            _, position = _varint(data, position)
        elif wire in (1, 5):
            position += 8 if wire == 1 else 4
        elif wire == 2:
            length, position = _varint(data, position)
            stop = position + length
            if stop > len(data):
                raise HarnessError("Truncated GP stylesheet message.")
            value = data[position:stop]
            selected = [path[1:] for path in paths if path[0] == field]
            if () in selected:
                replacement, count = b"", int(bool(value))
            elif selected:
                replacement, count = _clear_text_fields(value, selected)
            else:
                replacement, count = value, 0
            if count:
                output.extend(data[start:tag_end] + _encode_varint(len(replacement)) + replacement)
                changed += count
                position = stop
                continue
            position = stop
        else:
            raise HarnessError(f"Unsupported GP stylesheet wire type: {wire}")
        if position > len(data):
            raise HarnessError("Truncated fixed-width GP stylesheet value.")
        output.extend(data[start:position])
    return bytes(output), changed


def clear_template_attribution(root, payloads):
    """Remove artist/arranger credit fields from generated copies, not source files."""
    changed = []
    for field in ATTRIBUTION_FIELDS:
        node = root.find(f"./Score/{field}")
        if node is not None and node.text:
            node.text = ""
            changed.append(f"Score/{field}")
    cleaned = []
    for info, content in payloads:
        if info.filename == "Content/BinaryStylesheet":
            content, fields = _binary_stylesheet(content)
            changed.extend(fields)
        elif info.filename.startswith("Content/Stylesheets/") and info.filename.endswith(".gpss"):
            # GP7/8: page style -> first-page header -> attribution slot -> text style -> text.
            content, count = _clear_text_fields(content, [(6, 1, field, 2, 2) for field in range(3, 9)])
            if count:
                changed.append(f"{info.filename}: {count} attribution fields")
        cleaned.append((info, content))
    return cleaned, changed
