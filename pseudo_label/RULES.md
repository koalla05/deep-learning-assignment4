# Labeling rules for pseudo-labels

Task: list every place (location) mentioned in a Ukrainian or Russian Telegram post. The competition
scores the *set of strings* per post, after lowercasing and dropping digits and one-letter words, and
only exact string matches count. So the string must be copied exactly as it appears in the text.

These rules were written from the README conventions and the team's annotation policy, without looking
at the labels of `data/labeling_sample.csv`, so that sample stays a fair check of the labeler.

## Output

For each post, two lists of strings, copied character for character from the text:

- `locations`: places you are confident about.
- `uncertain`: places that match a "maybe" rule below. Scoring is done with and without them, and the
  better variant is kept.

Keep the original inflection and capitalization ("Києві", not "Київ"). Do not include the preposition
in front ("у Львові" -> "Львові"). Do not include trailing punctuation, quotes or emoji. Each distinct
string once per post is enough; repeated identical strings do not change the score.

## Label (put in `locations`)

- Countries, regions, oblasts, cities, towns, villages, settlements: "Україна", "Київщині", "Харкова",
  "Бучі", "Москве".
- Region and district phrases with their type word: "Київській області", "Оболонському районі",
  "Одеського району". Short forms as written: "обл." stays if it is part of the phrase.
- Streets, avenues, squares, lanes, boulevards, embankments **with the type word**, and with the house
  number if it follows: "вулиці Хрещатик", "вул. Січових Стрільців, 12", "проспекті Перемоги",
  "Майдані Незалежності", "бульвар Шевченка".
- City districts, microdistricts and neighbourhoods: "Троєщині", "Позняках", "Оболоні".
- Metro stations with the word if present: "станції метро Лук'янівська"; without it, the name alone.
- Physical geographic objects: rivers, seas, lakes, islands, mountains: "Дніпро" (the river), "Чорному морі".
- Highways and roads named by place: "трасі Київ – Чоп" -> label as written.

## Maybe (put in `uncertain`)

- Place names inside a channel signature, footer, subscription line or link text
  (e.g. a last line "Київ — пряма трансляція", "Підписатися | Одеса онлайн").
- Named buildings, bridges, parks, stations, airports, plants, malls and other landmarks used as a place:
  "Шулявського шляхопроводу", "парку Наталка", "аеропорту Бориспіль".
- Adjacent ambiguous cases where you are unsure whether a word is a place at all.

## Do not label

- Adjectives and demonyms on their own: "київський", "киян", "українські".
- Organization, company, team and media names, even if they contain a place: "Київенерго",
  "Укрзалізниця", "Динамо Київ" (the club), "Київська міська рада" (the institution).
- People, events, holidays, products.
- Generic words without a name: "місто", "вулиця", "район", "центр".

## Format of the answer file

A JSON list, one object per post, in the input order:

```json
[{"id": "t0", "locations": ["Київ"], "uncertain": []}]
```

## Adjudication (after labeling)

The batches were labeled by five separate agents. A consistency pass (`tools.adjudicate`) then made
the same decision everywhere for strings they disagreed on. It was written after the blind score on
the sample was measured and after looking at the sample's gold labels:

- Street names without a type word ("Межовій", "Хрещатику 36") are locations, as the rules above say.
- Countries and cities left in `uncertain` by mistake ("Білорусі", "Краків", "РФ") are locations.
- Coordinated names sharing one plural head are split into bare names, as in the sample's gold:
  "Луганських та Донецьких областей" -> "Луганських", "Донецьких"; "вулиць Котляревського та
  Северинівської" -> "Котляревського", "Северинівської"; leftover bare district adjectives are locations.
- Channel footers, place names inside organization names, front directions ("Лиманському напрямку")
  and landmarks stay in `uncertain` and are not used. On the sample, landmarks matched the gold 1 time in 4.
