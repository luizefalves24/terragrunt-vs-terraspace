#!/usr/bin/env python3
"""Classifica imagens do WhatsApp com CLIP (zero-shot) e separa as de "bom dia".

Duas etapas, para poder ajustar os limiares sem rodar o modelo de novo:

    python classify.py scan       whatsapp_images/ --out results.csv [--ocr]
    python classify.py quarantine whatsapp_images/ --csv results.csv --to quarantine/

`scan` grava em results.csv a probabilidade de cada categoria por imagem.
`quarantine` move para quarantine/<categoria>/ as imagens das categorias-alvo
acima do limiar e gera quarantine/review.html para revisão.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import shutil
import sys
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# Cada categoria tem vários prompts; a pontuação da categoria é o melhor deles.
# As categorias "de manter" existem para que fotos pessoais tenham para onde ir:
# zero-shot só funciona bem quando as alternativas cobrem o que aparece de verdade.
CATEGORIES: dict[str, list[str]] = {
    # --- alvos (vão para a quarentena) ---
    "saudacao": [
        "a 'good morning' greeting card image with decorative text",
        "a 'good night' greeting card image with moon and stars",
        "an image with the Portuguese text 'bom dia'",
        "an image with the Portuguese text 'boa noite'",
        "a religious blessing message with decorative text",
        "an inspirational quote over a glittery decorative background",
    ],
    "anjo": [
        "an illustration of an angel with wings",
        "a cute cartoon angel",
        "a picture of a cherub angel figurine",
    ],
    "flores_desenho": [
        "a drawing of flowers",
        "a clipart illustration of flowers",
        "a digital painting of a bouquet of flowers",
    ],
    "bebe": [
        "a stock photo of a cute baby",
        "a photo of a smiling baby",
    ],
    # --- manter ---
    "pessoal": [
        "a candid photo of people",
        "a selfie",
        "a family photo",
        "a group photo of friends",
    ],
    "flores_foto": [
        "a photograph of real flowers in a garden",
        "a close-up photograph of a real plant",
    ],
    "moto": ["a photo of a motorcycle", "a group of people riding motorcycles"],
    "documento": [
        "a photo of a document, receipt or paper",
        "a screenshot of a phone screen",
        "a screenshot of a chat conversation",
    ],
    "outros": [
        "a photo of food",
        "a landscape photograph",
        "a photo of a pet",
        "a funny meme",
        "a photo of a product or object",
        "a photo of a car",
    ],
}
TARGETS = ("saudacao", "anjo", "flores_desenho", "bebe")
CATEGORY_NAMES = list(CATEGORIES)

GREETING_RE = re.compile(
    r"\b(bom\s+dia|boa\s+(noite|tarde)|feliz\s+(domingo|segunda|ter[cç]a|quarta|quinta|sexta|s[aá]bado)"
    r"|aben[cç]oad[oa]|deus\s+(te|nos|vos)\s+aben[cç]oe|gratid[aã]o)\b",
    re.IGNORECASE,
)

# Recebe imagens RGB e devolve, para cada uma, {categoria: probabilidade}.
Classifier = Callable[[list[Image.Image]], list[dict[str, float]]]


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def load_rgb(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        return None


def make_clip_classifier(model_name: str) -> Classifier:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    prompts = [p for cat in CATEGORY_NAMES for p in CATEGORIES[cat]]
    owner = torch.tensor([i for i, cat in enumerate(CATEGORY_NAMES) for _ in CATEGORIES[cat]])

    def classify(images: list[Image.Image]) -> list[dict[str, float]]:
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image.cpu()  # (imagens, prompts)
        # Melhor prompt de cada categoria, depois softmax entre categorias:
        # assim categorias com mais prompts não levam vantagem.
        per_cat = torch.stack(
            [logits[:, owner == i].max(dim=1).values for i in range(len(CATEGORY_NAMES))], dim=1
        )
        probs = per_cat.softmax(dim=1)
        return [dict(zip(CATEGORY_NAMES, row.tolist())) for row in probs]

    return classify


def ocr_text(image: Image.Image) -> str:
    import pytesseract

    return " ".join(pytesseract.image_to_string(image, lang="por").split())


def scan(
    root: Path,
    out_csv: Path,
    classify: Classifier,
    batch_size: int = 32,
    ocr: Callable[[Image.Image], str] | None = None,
    ocr_min_target: float = 0.15,
) -> int:
    """Classifica as imagens de root e acrescenta em out_csv. Retoma de onde parou."""
    fields = ["path", "category", "score", *CATEGORY_NAMES, "ocr_hit", "ocr_text"]
    done: set[str] = set()
    if out_csv.exists():
        with out_csv.open(newline="", encoding="utf-8") as f:
            done = {row["path"] for row in csv.DictReader(f)}

    todo = [p for p in iter_images(root) if p.relative_to(root).as_posix() not in done]
    total = len(todo)
    new_file = not out_csv.exists()
    written = 0

    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for start in range(0, total, batch_size):
            batch = [(p, load_rgb(p)) for p in todo[start : start + batch_size]]
            batch = [(p, im) for p, im in batch if im is not None]
            if not batch:
                continue
            for (path, image), probs in zip(batch, classify([im for _, im in batch])):
                category = max(probs, key=probs.get)
                text = ""
                if ocr and sum(probs[t] for t in TARGETS) >= ocr_min_target:
                    text = ocr(image)
                writer.writerow(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "category": category,
                        "score": f"{probs[category]:.4f}",
                        **{c: f"{probs[c]:.4f}" for c in CATEGORY_NAMES},
                        "ocr_hit": int(bool(GREETING_RE.search(text))),
                        "ocr_text": text[:200],
                    }
                )
                written += 1
            f.flush()
            print(f"\r{min(start + batch_size, total)}/{total}", end="", file=sys.stderr)
    print(file=sys.stderr)
    return written


def pick_target(row: dict[str, str], thresholds: dict[str, float]) -> tuple[str, float] | None:
    """Categoria-alvo da imagem, ou None se ela deve ficar."""
    if row.get("ocr_hit") == "1":
        return "saudacao", 1.0
    category = row["category"]
    score = float(row[category])
    if category in TARGETS and score >= thresholds[category]:
        return category, score
    return None


def quarantine(root: Path, csv_path: Path, dest: Path, thresholds: dict[str, float]) -> dict[str, int]:
    """Move os alvos para dest/<categoria>/<caminho original> e escreve dest/review.html."""
    picked: dict[str, list[tuple[str, float]]] = {t: [] for t in TARGETS}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hit = pick_target(row, thresholds)
            if not hit:
                continue
            src = root / row["path"]
            if not src.exists():
                continue
            target = dest / hit[0] / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, target)
            picked[hit[0]].append((row["path"], hit[1]))

    write_review(dest, picked)
    return {cat: len(items) for cat, items in picked.items()}


def write_review(dest: Path, picked: dict[str, list[tuple[str, float]]]) -> None:
    # Menor pontuação primeiro: as dúvidas aparecem no topo de cada seção.
    sections = []
    for cat, items in picked.items():
        cards = "".join(
            f'<figure><img loading="lazy" src="{html.escape(cat + "/" + rel)}">'
            f"<figcaption>{html.escape(rel)}<br>{score:.2f}</figcaption></figure>"
            for rel, score in sorted(items, key=lambda x: x[1])
        )
        sections.append(f"<h2>{cat} ({len(items)})</h2><div class=grid>{cards}</div>")
    (dest / "review.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Quarentena</title><style>"
        "body{font-family:sans-serif;margin:16px}.grid{display:flex;flex-wrap:wrap;gap:8px}"
        "figure{margin:0;width:160px;font-size:11px;word-break:break-all}"
        "img{width:160px;height:160px;object-fit:cover}</style>"
        "<h1>Revisar quarentena</h1><p>Para manter uma imagem, tire-a da pasta da quarentena. "
        "O que ficar na pasta será apagado do celular por delete_on_phone.sh.</p>" + "".join(sections),
        encoding="utf-8",
    )


def parse_thresholds(default: float, overrides: list[str]) -> dict[str, float]:
    thresholds = {t: default for t in TARGETS}
    thresholds["bebe"] = max(default, 0.75)  # fotos de bebê da família são o maior risco
    for item in overrides:
        name, _, value = item.partition("=")
        if name not in TARGETS:
            raise SystemExit(f"categoria desconhecida: {name} (use {', '.join(TARGETS)})")
        thresholds[name] = float(value)
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="classifica as imagens e grava o CSV")
    s.add_argument("root", type=Path)
    s.add_argument("--out", type=Path, default=Path("results.csv"))
    s.add_argument("--model", default="openai/clip-vit-base-patch32")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--ocr", action="store_true", help="confirma texto 'bom dia' etc. com Tesseract")

    q = sub.add_parser("quarantine", help="move os alvos para a quarentena")
    q.add_argument("root", type=Path)
    q.add_argument("--csv", type=Path, default=Path("results.csv"))
    q.add_argument("--to", type=Path, default=Path("quarantine"))
    q.add_argument("--threshold", type=float, default=0.5)
    q.add_argument("--set", action="append", default=[], metavar="CAT=VALOR", help="ex.: --set bebe=0.9")

    args = parser.parse_args()
    if args.cmd == "scan":
        n = scan(
            args.root,
            args.out,
            make_clip_classifier(args.model),
            batch_size=args.batch_size,
            ocr=ocr_text if args.ocr else None,
        )
        print(f"{n} imagens classificadas -> {args.out}")
    else:
        counts = quarantine(args.root, args.csv, args.to, parse_thresholds(args.threshold, args.set))
        for cat, n in counts.items():
            print(f"{cat:>15}: {n}")
        print(f"Revise em {args.to / 'review.html'}")


if __name__ == "__main__":
    main()
