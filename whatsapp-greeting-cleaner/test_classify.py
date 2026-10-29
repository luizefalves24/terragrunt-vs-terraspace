"""Testa o fluxo scan -> quarantine com um classificador falso (sem torch)."""

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import classify as c


def fake_classifier(images):
    # Imagens vermelhas = saudação, verdes = bebê (pontuação 0.6), azuis = pessoal.
    out = []
    for im in images:
        r, g, b = im.getpixel((0, 0))
        probs = dict.fromkeys(c.CATEGORY_NAMES, 0.0)
        if r > 200:
            probs["saudacao"] = 0.9
        elif g > 200:
            probs["bebe"] = 0.6
        else:
            probs["pessoal"] = 0.8
        out.append(probs)
    return out


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "imgs"
        (self.root / "Sent").mkdir(parents=True)
        Image.new("RGB", (8, 8), (255, 0, 0)).save(self.root / "IMG-1-WA0001.jpg")
        Image.new("RGB", (8, 8), (255, 0, 0)).save(self.root / "Sent" / "IMG-1-WA0001.jpg")
        Image.new("RGB", (8, 8), (0, 255, 0)).save(self.root / "IMG-2-WA0002.jpg")
        Image.new("RGB", (8, 8), (0, 0, 255)).save(self.root / "IMG-3-WA0003.jpg")
        (self.root / "corrompida.jpg").write_bytes(b"not an image")
        (self.root / ".nomedia").write_bytes(b"")
        self.csv = self.tmp / "results.csv"

    def test_scan_resume_and_quarantine(self):
        self.assertEqual(c.scan(self.root, self.csv, fake_classifier, batch_size=2), 4)
        self.assertEqual(c.scan(self.root, self.csv, fake_classifier), 0)  # retoma: nada novo

        with self.csv.open() as f:
            rows = {r["path"]: r for r in csv.DictReader(f)}
        self.assertEqual(rows["Sent/IMG-1-WA0001.jpg"]["category"], "saudacao")

        dest = self.tmp / "quarantine"
        counts = c.quarantine(self.root, self.csv, dest, c.parse_thresholds(0.5, []))
        # bebê com 0.6 fica abaixo do limiar padrão de bebê (0.75)
        self.assertEqual(counts, {"saudacao": 2, "anjo": 0, "flores_desenho": 0, "bebe": 0})
        self.assertTrue((dest / "saudacao" / "Sent" / "IMG-1-WA0001.jpg").exists())
        self.assertTrue((self.root / "IMG-2-WA0002.jpg").exists())
        self.assertTrue((self.root / "IMG-3-WA0003.jpg").exists())
        self.assertIn("Sent/IMG-1-WA0001.jpg", (dest / "review.html").read_text())

    def test_ocr_hit_wins(self):
        c.scan(self.root, self.csv, fake_classifier, ocr=lambda im: "Bom Dia! Deus te abençoe")
        with self.csv.open() as f:
            rows = {r["path"]: r for r in csv.DictReader(f)}
        self.assertEqual(rows["IMG-2-WA0002.jpg"]["ocr_hit"], "1")  # bebê 0.6 -> OCR rodou
        self.assertEqual(rows["IMG-3-WA0003.jpg"]["ocr_hit"], "0")  # pessoal -> OCR pulado
        self.assertEqual(c.pick_target(rows["IMG-2-WA0002.jpg"], c.parse_thresholds(0.5, [])), ("saudacao", 1.0))

    def test_threshold_override(self):
        self.assertEqual(c.parse_thresholds(0.5, ["bebe=0.55"])["bebe"], 0.55)
        with self.assertRaises(SystemExit):
            c.parse_thresholds(0.5, ["gato=0.1"])


if __name__ == "__main__":
    unittest.main()
