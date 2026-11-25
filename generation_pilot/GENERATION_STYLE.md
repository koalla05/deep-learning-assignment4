# Generation style, version 0.1

Execution update: the user subsequently chose `gpt-6.1-terra` with high reasoning
through subagents and parent-agent review, superseding the local-generator
recommendation below. There are now 50 posts and a completed exploratory
[AUC report](AUC_REPORT.md). The following records the original recipe
and pre-generation inspection, whose fingerprint was frozen before drafting.

Recipe status at initial inspection: proposed for review, before generation.
The user confirmed that synthetic supervision and aggregate test-text adaptation
are allowed. The user can review posts; a second human reviewer is not yet
available. The existing double-review release gate remains unmet. Runtime notes
below describe the pre-generation inspection; the AUC report records the later
classifier environment and completed fits.

## Evidence and scope

Generate original Ukrainian/Russian Telegram-style local-news posts with exact
location spans. Match the distribution of writing, not specific test events,
wording, channel identities, or place-name inventories. Follow PLAN.md for the
1,000-attempt ceiling, 600–800 accepted-post aim, protected fold 0, parent/family
splits, review requirements, and optional subsequent AUC experiment.

The immediate next step is a reviewable Stage A batch, not classifier training.
The table below describes all 477 test posts, inspected before a new target split.
This is disclosed target exposure, not a clean held-out profile. Before generation,
freeze target groups and obtain any further recipe statistics from the fit side
only. Do not use new evaluation-side results to adjust this recipe.

| Property | Observed | Generation decision |
| --- | --- | --- |
| Character length, p25 / median / p75 | 128 / 264 / 517 | Mix terse alerts and several-paragraph posts |
| Character length, mean / p95 / maximum | 395.4 / 1,123.2 / 3,682 | Preserve a small long tail; do not pad every post |
| Contains a line break | 85.1% | Aim for about 85% across a batch |
| Contains a blank line | 79.9% | Aim for about 80%; vary paragraph placement |
| Contains emoji, using the notebook's broad range regex | 60.2% | Aim for about 60%; include plain posts too |
| Contains an explicit URL | 4.0% | Mostly omit; use inert reserved example-domain URLs if needed |
| Contains an @handle | 4.2% | Initially omit rather than invent real-looking accounts |
| Exact unique texts | 477 / 477 | Require generated uniqueness and check near duplicates separately |
| XLM-R tokens, median / p95 / max, existing EDA | 73 / 314 / 926 | Measure with the real tokenizer before release |

Line-break, emoji, and URL percentages are marginal summaries, not independent
coin flips or exact quotas. Respect natural combinations for each genre.
Language is approximately 90% Ukrainian / 10% Russian, following the existing
plan; script heuristics do not establish exact language prevalence.

The 26 labeled examples have 32 annotated occurrences and 10 empty-label posts.
Every supplied span reconstructs its supplied string, but this does not establish
completeness. A route post labels only its first two locations despite later
station names; a long notice omits earlier city mentions; an address span cuts
off its last digit. Do not learn these omissions or truncations as rules. Do not
infer the target's empty-post rate or mention-count distribution from this sample.

## Writing brief

Write as a local editor or resident sharing a useful update. Open with the event,
change, observation, or question. Give a few concrete details: a time window,
route change, service interruption, work duration, measurement, or next action.
Keep one coherent subject. Use ordinary Ukrainian or Russian, natural inflection,
and occasional informal phrasing. Avoid the encyclopedic and literary tone of
many source sentences.

Vary the layout: a single line; headline and short body; two or three paragraphs;
bullet or measurement list; short quoted notice; or question followed by an
update. Some posts have a concise call to action. Do not put a headline, emoji,
list, quote, and footer into every post. Do not repeat unrelated source sentences
to reach a length target.

Allow ordinary punctuation variation, Ukrainian apostrophes, abbreviations,
parentheses, times, route numbers, and occasional uppercase emphasis. Start with
grammatical text; do not inject systematic typos or character noise. Numbers in
the narrative are not automatically locations. Emoji should serve the message,
usually zero to three, rather than decorate every sentence.

Use new neutral footers, sparingly, such as an invitation to share an update.
Do not copy channel signatures, handles, URLs, distinctive slogans, or specific
events from the competition data. Initially keep footer text free of locations
and named organizations so footer annotation does not require a guessed rule.

## Proposed sampling recipe

These allocations are design choices, not measured topic or label proportions.
Freeze them before generation and results; record deviations caused by rejection.

| Genre | Share | Content and layout variation |
| --- | --- | --- |
| Transport and traffic | 25% | Delays, changed stops, routes, timetables, short resident observations |
| Roadworks and utilities | 20% | Repairs, access restrictions, service windows, completion updates |
| Municipal notices | 20% | Opening hours, maintenance, service announcements, quoted notices |
| Community and local events | 15% | Small events, neighborhood updates, questions, follow-up messages |
| Weather and measurement lists | 10% | Rain, temperatures, readings with units; no copied target lists |
| General brief updates | 10% | Useful announcements, including genuinely location-free messages |

Across genres, aim for about 15% of candidates below 80 characters, 30% at
80–220, 30% at 221–500, 20% at 501–1,100, and 5% at 1,101–3,700. These broad
starting buckets approximate the observed spread; they are not a fitted
histogram. Never truncate a completed post to satisfy a bucket. Check actual
token lengths, including the small tail above 512 tokens.

For annotation diversity, start with 15% location-free candidates, 45% with one
occurrence, 25% with two or three, 10% with four to six, and 5% with seven or more.
This is a proposed training mix, not inferred test gold. Repeated occurrences
count separately. Include lists, repeated names, nearby adjacent names, and
multiword locations without making every post a dense gazetteer listing.

Stage A: at most 150 attempts, approximately half conservative assembly and half
location-first generation; use the 90/10 language allocation and cover all six
genres. Start by showing the user 20 stratified candidates from that budget for
fluency and policy feedback. Audit the entire stage, then permit at most one
recipe revision as PLAN.md requires. The 20 are not extra attempts. Do not label
agent-reviewed candidates as human accepted.

## Proposed annotation policy

Use this consistent policy for new data, subject to the user's review of Stage A.
It resolves ordinary cases without pretending the sample provides an exhaustive
annotation manual. Quarantine remaining ambiguous constructions.

| Case | Proposed action |
| --- | --- |
| City, settlement, country, region, neighborhood used geographically | Annotate the exact surface form, including grammatical inflection |
| Street with a type word | Include the type word and full name: `вулиці Каштановій`; exclude the preceding preposition |
| Full address with a building number | Include the internal comma and entire number, including a suffix if present; exclude trailing sentence punctuation |
| Explicit district phrase | Include the complete contiguous phrase when present, such as adjective plus `районі` |
| Coordinated district adjectives sharing a head noun | Initially avoid; do not invent a noncontiguous span or alternate boundary convention |
| Named station, bridge, park, mall, or building as a physical destination | Annotate the full approved place phrase; quarantine uncertain name/category boundaries |
| Organization title containing a geographic adjective or name | Do not extract a nested place solely because its spelling occurs inside the organization title |
| Organization versus physical venue ambiguity | Avoid or quarantine until adjudicated; approved physical-place contexts must be explicit |
| Geographic adjective used only descriptively, demonym, generic `місто` | Do not label on spelling alone; prefer unambiguous examples initially |
| Place in a quoted event description | Apply the same geographic-use rule as ordinary body text |
| Place in channel branding, footer, handle, or URL | Avoid these constructions initially; sample omissions do not establish a general rule |
| Same location repeated in body text | Annotate every geographic occurrence, even if surface forms repeat |
| Two adjacent names | Separate spans, each covering its own mention; do not merge across conjunctions |
| Capitalization, inflection, punctuation | Preserve the final text exactly; no lemmatization or scorer-driven editing of gold |
| No-location post | Empty spans only after reviewing the complete text, including its footer |

Annotate all unambiguous body locations, including incidental mentions outside the
main event. A deliberate missing label is not a way to mimic inconsistent sample
annotations. Keep uncertain cases out of accepted training data.

Offsets are Python Unicode code-point indices, half open `[start, end)`, on the
final text. They are not UTF-8 byte offsets or browser UTF-16 offsets. Derive
`locations` from spans; preserve occurrence order and repeated occurrences in
the canonical candidate record. Apply the official metric's deduplication only
when scoring. Preserve internal address punctuation even though score-time
normalization is imperfect.

## Generation mechanics and local resources

Actual `cv_rows.parquet` schema: `source`, `row_id`, `doc_id`, `group_id`,
`role_v2`, `text_key`, `has_location`, `fold`, `text`, `loc_markers`.
Markers are serialized Python literal strings, not already parsed lists.
There are 61,660 `train` and 1,728 `development` rows, with no `exclude_*` or
`final` roles in this file. Preserve grouped CV eligibility rather than applying
a new split to these roles. Folds 1 and 2 contain 42,259 eligible parent rows
across 22,612 groups; fold 0 remains protected. A group's rows occupy one fold.

Conservative assembly: audit parents first, then combine only related sentences
with compatible language, place context, and event details. Make minimal
connective edits through an explicit edit map. If coherent assembly is impossible,
reject instead of stitching unrelated facts together. Audit mentions again after
editing; original source labels are not automatically complete.

Location-first generation: collect approved names and attested inflected forms
only from eligible training-side resources. Require documented provenance for
additional resources. Assign domain pools to parents/resources and template
families before use; carry location-resource parent IDs into derivatives too.
Names may occur independently in different pools, but parent lineage must not
cross pools. Use no more than three candidates per parent group.

Select a coherent set of places, their geographic roles, grammatical frames,
genre, length bucket, and pool. Ask the generator for a placeholder draft only:

```text
Write one original {language} Telegram-style post.
Genre: {genre}. Topic: {invented_event_brief}.
Layout: {layout}. Length: {character_bucket}.
Use exactly these unique location slots: {slot_specifications}.
Each slot has a grammatical frame and geographical role; use it once.
Introduce no other named places, organizations, accounts, or channel identities.
Do not change slot identifiers. Use natural, varied prose and one coherent event.
Do not add an explanation or annotations. Return only the draft.
```

The generator receives no target post or labeled sample text. For a repeated
place, give each occurrence a distinct slot. Insert reviewed, grammatically
compatible surface forms through one assembler, calculating spans as text is
appended. Validate final lengths after insertion. Reject unresolved slots,
unplanned places, bad inflection, contradictory details, and protected-text
overlap. Checking slots alone cannot establish label completeness.

Ollama is installed. Local cache manifests exist for `qwen3.5:4b` and
`gpt-oss:20b`; a separate `gpt-oss:20b-cloud` manifest is not a local-generation
option. Proposed first choice: check the cached 4B model's availability and
Ukrainian/Russian fluency on a small Stage A subset. Manifest presence does not
prove runnable weights, sufficient memory, or quality. Do not download weights,
start the larger model, or use a cloud fallback automatically. If local fluency
fails, report that rather than silently expanding compute scope.

The current default Python lacks pandas, pyarrow, torch, and transformers.
The sibling assignment 3 environment has pandas, pyarrow, and torch; it was used
read-only to inspect the parquet schema. Transformers was absent there. No
packages were installed. A reproducible runtime remains an implementation task.

## Review and acceptance

Export candidate text with highlighted spans, extracted strings, language, genre,
parents, policy version, and reviewer fields. The user reviews the entire text
for missing/extra mentions, boundaries, grammar, and coherence. Keep `pending`,
`accepted`, and `rejected` explicit. Reviewer corrections must update spans and
be validated on the final text.

The existing plan additionally requires independent annotation of at least 30
Stage A candidates and double review of at least 50 accepted posts overall.
No second reviewer is available yet. This does not prevent drafting or primary
review, but do not claim completion of the pilot or proceed through its review
release gate without satisfying that requirement or explicitly revising it.

Unresolved decisions are now limited to assigning the second reviewer and
adjudicating any concrete ambiguous examples found during Stage A. The proposed
ordinary annotation rules and sampling choices above can be reviewed together
with actual candidates instead of requiring an abstract questionnaire.

## Input fingerprints

SHA-256, measured during this inspection:

```text
data/test.csv
8cd69906e64b9626d85a9fcb33df4e6b0d7c3d93290c1b562e4586878688b60d
data/labeling_sample.csv
e1c9ab5022422c1ecfafb62c2d47a535078a3271fcdb5d00e9a9fbc25ce2653e
data/cv_rows.parquet
aa206c43385e370094d0e6e4059cacdd414d2c6a74f533e88ef795863f4cd104
```

Measurements used Python CSV parsing, character counts, linear-interpolated
quantiles, and the existing notebook's URL/emoji regexes; no new tokenizer run
was performed. Sources: `data/`, the existing EDA/adversarial notebooks, and
the assignment 3 split policy. No external data, final labels, generated posts,
model fits, or submissions were used.
