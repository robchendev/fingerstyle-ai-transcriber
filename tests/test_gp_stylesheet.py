from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipInfo

from scripts.dataset_io import ROOT
from scripts.gp_normalization import _parse_archive, _archive_bytes
from scripts.gp_output import write_gp_outputs
from scripts.gp_stylesheet import _binary_stylesheet, _clear_text_fields, _encode_varint, clear_template_attribution
from scripts.transcriber_audio import HarnessError
from tests.test_gp_normalization import archive_bytes
from tests.test_gp_output import output_template, hypotheses


def field(number, value):
    return _encode_varint(number << 3 | 2) + _encode_varint(len(value)) + value


def stylesheet(credit):
    return field(6, field(1, field(5, b"\x08\x01" + field(2, field(1, b"font") + field(2, credit) + b"\x18\x01")))) + field(77, b"unknown-style-resource")


def binary(credit):
    key = b"Header/Music"
    item = bytes([len(key)]) + key + b"\x03" + len(credit).to_bytes(2, "big") + credit
    preserved = b"\x09Font/Size\x01\x00\x00\x00\x11"
    return (2).to_bytes(4, "big") + item + preserved


class GPStylesheetTests(unittest.TestCase):
    def test_only_attribution_changes_and_original_styles_are_preserved(self):
        root = ET.fromstring("<GPIF><Score><Artist>Template performer</Artist><Music>Template arranger</Music><Copyright>Retain notice</Copyright><Title>Title</Title></Score><Notes/></GPIF>")
        before = ET.tostring(root)
        credit = b"Arranged by a template author with a long credit " * 4
        payloads = [(ZipInfo("Content/BinaryStylesheet"), binary(credit)),
                    (ZipInfo("Content/Stylesheets/score.gpss"), stylesheet(credit)),
                    (ZipInfo("future/resource"), b"preserve all bytes")]
        copied = deepcopy(root)
        cleaned, changed = clear_template_attribution(copied, payloads)
        self.assertEqual(ET.tostring(root), before)
        self.assertEqual(copied.findtext("./Score/Artist"), "")
        self.assertEqual(copied.findtext("./Score/Copyright"), "Retain notice")
        self.assertEqual(copied.findtext("./Score/Title"), "Title")
        self.assertEqual(cleaned[0][1], binary(b""))
        self.assertEqual(cleaned[1][1], stylesheet(b""))
        self.assertEqual(cleaned[2], payloads[2])
        self.assertEqual(len(changed), 4)
        repeated, changes = clear_template_attribution(copied, cleaned)
        self.assertEqual(repeated, cleaned)
        self.assertEqual(changes, [])

    def test_truncated_or_unknown_encodings_fail_instead_of_corrupting_resources(self):
        for data in (b"", b"\x00\x00\x00\x01", binary(b"abc")[:-1]):
            with self.subTest(data=data), self.assertRaises(HarnessError):
                _binary_stylesheet(data)
        for data in (b"\x00", b"\x0a\x05a", b"\x0d\x01", b"\x80" * 10):
            with self.subTest(data=data), self.assertRaises(HarnessError):
                _clear_text_fields(data, [(6, 1, 5, 2, 2)])

    def test_both_export_variants_strip_inherited_stylesheet_credits(self):
        root, payloads, comment = _parse_archive(archive_bytes(output_template()))
        replacements = {"Content/BinaryStylesheet": binary(b"Template musician"),
                        "Content/Stylesheets/score.gpss": stylesheet(b"Template musician")}
        raw = _archive_bytes(root, [(info, replacements.get(info.filename, value)) for info, value in payloads], comment)
        with TemporaryDirectory(dir=ROOT) as folder:
            folder = Path(folder)
            template = folder / "source.gpt"
            template.write_bytes(raw)
            report = write_gp_outputs(template, hypotheses(), folder / "full.gp", folder / "single.gp")
            self.assertEqual(template.read_bytes(), raw)
            for variant in ("fullVoices", "singleVoice"):
                self.assertTrue(report[variant]["clearedTemplateAttribution"])
                _, resources, _ = _parse_archive(Path(report[variant]["path"]).read_bytes())
                for info, value in resources:
                    if info.filename in replacements:
                        self.assertNotIn(b"Template musician", value)
