"""Build the Distil Labs input folder from the pseudo-labeled test posts.

Distil Labs has no NER task, so each post becomes a question-answering example: the post is the user
message and a JSON list of its places is the answer. Seed half A is the train split (what synthetic
data is generated from) and half B is the test split, so we can later check generalization on B.

Run from the repo root:  python synthetic/prepare_distil.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "distil_input"

TASK = """Extract every place (location) mentioned in a Ukrainian or Russian Telegram post from Kyiv local news channels.
Answer with a JSON list of strings, each copied exactly as it appears in the post (same inflection, capitalization and spelling), without the preceding preposition and without trailing punctuation. List each distinct string once. Answer [] if there are no places.
Label: countries, regions, oblasts, cities, towns, villages, city districts and neighbourhoods, streets/avenues/squares with their type word and house number when present ("вулиці Хрещатик", "вул. Січових Стрільців, 12"), metro stations, rivers and seas, and district phrases ("Оболонському районі"). Coordinated names sharing one plural head are listed bare ("Луганських та Донецьких областей" -> "Луганських", "Донецьких").
Do not label: adjectives and demonyms ("київський", "киян"), organization or media names even if they contain a place ("Київенерго", "Національна поліція України"), place names inside channel signatures or footers ("Київ — пряма трансляція"), landmarks such as malls, parks, monuments and hospitals, and front directions ("Лиманському напрямку")."""

JUDGE = """The answer must be a JSON list of strings. It is correct when it contains exactly the places present in the post according to the task description: every string must occur verbatim in the post, no place may be missing, and no organization name, footer, landmark or adjective may be included. The order of strings does not matter."""

SYNTH = """Write new, realistic Telegram posts in the style of Kyiv local news channels, like the examples. Use line breaks and occasional channel-style footers as in the examples; about half of the posts use no emoji at all, the rest one to three. Write fluent, grammatical Ukrainian or Russian. Vary the places: many different Kyiv districts, neighbourhoods, streets with house numbers or number ranges, metro stations, villages and towns of Kyiv oblast and other regions of Ukraine, and foreign countries; avoid reusing the same few places. Do not copy or paraphrase the example posts and do not reuse their structure; invent new events, people, numbers and wording, and start posts in different ways. Label each new post exactly following the task description; an address like "вулиці Кірова, 2-14" keeps its numbers."""

# Diversity controls (Distil "mutators"). Each generation call samples one value per mutator; weights
# follow the seed posts: lengths from the test quartiles, about a third of posts with no place, 90/10
# Ukrainian/Russian. The smoke run without them gave short, emoji-heavy, place-dense, repetitive posts.
MUTATORS = """  mutators:
    - name: length
      values:
        - "very short: one line under 80 characters"
        - "short: 80 to 220 characters, one or two short paragraphs"
        - "medium: 220 to 500 characters, two or three paragraphs"
        - "long: 500 to 1100 characters, several paragraphs or a list"
        - "very long: 1100 to 2500 characters, a detailed multi-paragraph report or long list"
      target_distribution: [15, 30, 30, 20, 5]
    - name: places
      values:
        - "without any place name at all, so the answer is an empty list"
        - "mentioning exactly one place, possibly repeated"
        - "mentioning two or three different places"
        - "mentioning four or more different places, for example a list of streets, districts or towns"
      target_distribution: [30, 35, 20, 15]
    - name: language
      values: ["written in Ukrainian", "written in Russian"]
      target_distribution: [9, 1]
    - name: topic
      values:
        - "about an air raid alert, a drone or missile attack, or its consequences"
        - "about power, water, heating or gas outages and utility works"
        - "about public transport, traffic, roadworks or a road accident"
        - "about city council or authorities' decisions, budgets and public services"
        - "about crime, police, courts or corruption"
        - "about events, culture, sport, schools or volunteering"
        - "about weather, environment or animals"
        - "about the war, the front, politics or international news"
        - "about prices, business, shops or the economy"
        - "about everyday city life: a resident's complaint, a question or a lost-and-found post"
"""


def main():
    pseudo = pd.read_parquet(ROOT.parent / "pseudo_label" / "output" / "test_pseudo.parquet")
    OUT.mkdir(exist_ok=True)
    for half, split in [("A", "train"), ("B", "test")]:
        rows = pseudo[pseudo["seed_half"] == half]
        with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows.itertuples():
                example = {
                    "messages": [
                        {"role": "user", "content": r.text},
                        {"role": "assistant", "content": json.dumps(list(r.locations), ensure_ascii=False)},
                    ]
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"{split}.jsonl: {len(rows)} posts (seed half {half})")

    job = {
        "task_description": TASK,
        "llm_as_a_judge_instructions": JUDGE,
        "synthetic_data_generation_instructions": SYNTH,
    }
    (OUT / "job_description.json").write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Proprietary models (Claude, GPT, Gemini) are not allowed as generation teachers on Distil Labs,
    # so GLM 5.3, the recommended open-weight teacher, writes the synthetic posts.
    config = """base:
  task: question-answering
  teacher_model_name: zai.glm-5.3-low-thinking
synthgen:
  output_is_json: true
  train_generation_target: {target}
  teacher_temperature: 0.7
  num_positive_exemplars_per_generation: 3
  generation_in_single_call: 2
  validation_similarity_threshold: 0.85
{mutators}  validation_instructions: |
    Read the post and extract the places yourself, following the task description.
    Reject the example if your list differs from the answer (a missing place, an extra string,
    or a string that does not occur verbatim in the post), if the answer is not a JSON list of
    strings, or if the post is not a plausible Ukrainian or Russian Telegram news post.
"""
    (OUT / "config.yaml").write_text(config.format(target=6000, mutators=MUTATORS), encoding="utf-8")
    # Smoke run: same settings, 128 rows, to check the labels before paying for the full run.
    (ROOT / "smoke_config.yaml").write_text(config.format(target=128, mutators=MUTATORS), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
