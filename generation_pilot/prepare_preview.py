"""Freeze preview inputs without exposing protected texts to generators.

Run with a Python environment providing pandas and pyarrow. No network calls.
"""

import ast
import csv
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "preview_001"
SEED = 42


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def pool(key):
    value = int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest(), 16)
    return "evaluation" if value % 5 == 0 else "fit"


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    import pandas as pd

    if OUT.exists():
        raise SystemExit("Refusing to overwrite a frozen preview directory.")
    OUT.mkdir(parents=True)
    data = ROOT.parent / "data"
    cv = pd.read_parquet(data / "cv_rows.parquet")
    eligible = cv[cv.fold.isin([1, 2])]
    assert not eligible.role_v2.str.startswith("exclude_").any()
    assert not eligible.role_v2.eq("final").any()
    assert cv.groupby("group_id").fold.nunique().max() == 1
    dump(
        "source_manifest.json",
        [
            {
                "parent_id": f"{r.source}:{r.row_id}",
                "source": r.source,
                "row_id": r.row_id,
                "group_id": r.group_id,
                "fold": r.fold,
                "role_v2": r.role_v2,
                "domain_pool": pool(r.group_id),
            }
            for r in eligible.itertuples()
        ],
    )

    # Conservative automated target grouping; keep suspicious reposts together.
    # No target text is exported. This is not a substitute for the full protocol's
    # human near-duplicate review before any AUC experiment.
    targets = list(csv.DictReader((data / "test.csv").open()))
    texts = [normalized(r["text"]) for r in targets]
    shingles = [{t[i : i + 5] for i in range(max(0, len(t) - 4))} for t in texts]
    parents = list(range(len(texts)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    edges = []
    for i, a in enumerate(shingles):
        for j in range(i):
            b = shingles[j]
            intersection = len(a & b)
            union = len(a | b)
            jaccard = intersection / union if union else 0
            containment = intersection / min(len(a), len(b)) if a and b else 0
            if texts[i] == texts[j] or (
                min(len(texts[i]), len(texts[j])) >= 40 and (jaccard >= 0.5 or containment >= 0.8)
            ):
                parents[find(i)] = find(j)
                edges.append(
                    {
                        "a": targets[i]["text_id"],
                        "b": targets[j]["text_id"],
                        "jaccard": round(jaccard, 4),
                        "containment": round(containment, 4),
                    }
                )
    groups = {}
    for i, r in enumerate(targets):
        groups.setdefault(find(i), []).append(str(r["text_id"]))
    keys = {
        root: "target:" + hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:16]
        for root, ids in groups.items()
    }
    dump(
        "target_manifest.json",
        [
            {"text_id": r["text_id"], "group_id": keys[find(i)], "domain_pool": pool(keys[find(i)])}
            for i, r in enumerate(targets)
        ],
    )
    dump("target_overlap_edges.json", edges)

    # Hand-selected grammatical surface forms, verified against training-side
    # occurrences. This list was not obtained from target texts or target labels.
    city_forms = {
        "Полтаві",
        "Чернігові",
        "Черкасах",
        "Сумах",
        "Вінниці",
        "Рівному",
        "Луцьку",
        "Ужгороді",
        "Кропивницькому",
        "Тернополі",
        "Чернівцях",
        "Хмельницькому",
        "Львові",
        "Одесі",
        "Дніпрі",
        "Харкові",
        "Запоріжжі",
        "Луцке",
        "Москве",
        "Рыбинске",
        "Киеве",
        "Одессе",
        "Харькове",
    }
    street_forms = {
        "вулиці Станіславського, 50",
        "проспекті Богдана Хмельницького",
        "вулиці Вітовського, 57",
        "вулиці Коломийській, 19",
        "вул. Винниченка",
        "вул. Соборності, 50",
        "вул. Соборності, 46-в",
        "вулиці Зеленій",
        "вулиці Тролейбусній",
        "вулиці Гожулянській",
        "вул. Троїцькій",
    }
    bank = []
    seen = set()
    for r in eligible.itertuples():
        for a, b in ast.literal_eval(r.loc_markers):
            surface = r.text[a:b]
            if surface not in city_forms | street_forms:
                continue
            key = (r.source, surface)
            if key in seen:
                continue
            seen.add(key)
            bank.append(
                {
                    "resource_id": f"loc_{len(bank) + 1:03}",
                    "surface": surface,
                    "language": r.source,
                    "kind": "street" if surface in street_forms else "city",
                    "grammatical_frame": (
                        "на" if surface in street_forms else ("в" if r.source == "ru" else "у/в")
                    ),
                    "parent_id": f"{r.source}:{r.row_id}",
                    "parent_group_id": r.group_id,
                    "parent_span": [a, b],
                    "domain_pool": pool(r.group_id),
                }
            )
    dump("surface_bank.json", bank)
    # Disjoint families and neutral closing banks across domain pools.
    families = {
        p: [
            f"{p}_{g}_{layout}"
            for g in ["transport", "roadworks", "municipal", "community", "weather", "general"]
            for layout in ["brief", "update", "list"]
        ]
        for p in ["fit", "evaluation"]
    }
    config = {
        "seed": SEED,
        "generator": "gpt-6.1-terra",
        "reasoning_effort": "high",
        "reviewer": "parent Codex agent; exact model not recorded",
        "candidate_budget": 20,
        "total_attempt_ceiling": 1000,
        "stage_A_attempt_ceiling": 150,
        "recipe_version": "0.2-terra-preview",
        "policy_version": "0.1",
        "allowed_cv_fold": 0,
        "families": families,
        "footer_banks": {"fit": [], "evaluation": []},
        "preview_stream": "location_first",
        "scope_note": "20 location-first previews; remaining Stage A must balance streams.",
        "target_near_duplicate_review": "automated conservative grouping; human review pending",
        "human_review_status": "pending",
        "tokenizer_validation": "pending",
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [data / "test.csv", data / "labeling_sample.csv", data / "cv_rows.parquet"]
        },
    }
    dump("frozen_config.json", config)
    digest = hashlib.sha256((ROOT / "GENERATION_STYLE.md").read_bytes()).hexdigest()
    dump("recipe_fingerprint.json", {"GENERATION_STYLE.md": digest})
    print(
        json.dumps(
            {
                "resources": len(bank),
                "target_groups": len(groups),
                "suspicious_edges": len(edges),
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
