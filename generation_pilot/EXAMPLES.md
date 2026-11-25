# Worked examples and failure cases

These are invented documentation examples, not a generated training dataset.
They are not copied or paraphrased from competition posts. Named surface forms
illustrate mechanics; the actual pilot must use approved training-side resources.

## A. Assemble labels, rather than guess offsets afterward

Structured input:

```text
language: Ukrainian
genre: municipal announcement
pattern: "У <LOC_1> відкрили новий маршрут."
LOC_1 surface form: "Черкасах"
LOC_1 meaning: city, geographical use
```

Illustrative final text:

```text
У Черкасах відкрили новий маршрут.
```

Gold occurrence span: `(2, 10)`; `text[2:10] == "Черкасах"`.
Keep the inflected surface form, not the dictionary form "Черкаси".
The reviewer must still confirm that no other location is present and the
sentence is plausible and grammatical.

For a multi-line post, insert all strings and separators through the same
assembler. Adding a headline, emoji, or footer after computing offsets without
updating them is a rejection.

## B. Lists and repeated mentions

```text
Дані за ранок:
<LOC_1> - 12
<LOC_2> - 9

У <LOC_3> вимірювання повторять увечері.
```

Choose grammatically appropriate forms for each slot. If LOC_1 and LOC_3 refer
to the same city, annotate both occurrences. The output string set may later
deduplicate equal surface forms; the occurrence labels must not.

Do not use global string replacement to label every matching substring.
Track each slot occurrence separately.

## C. Street-prefix boundaries

```text
Рух на <LOC_1> тимчасово обмежено.
```

Possible approved insertion:

```text
вулиці Каштановій
```

The generated span may cover the whole insertion if the frozen policy includes
street-type words in this construction. Do not label only "Каштановій" in one
family and the full phrase in another without a documented contextual reason.

The supplied sample includes full street-prefix spans, but is too small and
inconsistent to justify adding every possible prefix blindly.

## D. Negative examples require exhaustive checks

Candidate:

```text
Завтра оновлять графік прийому. Деталі повідомимо вранці.
```

An empty label list is plausible after review.

Counterexample:

```text
Завтра оновлять графік прийому. Новини Черкас - підписатися.
```

This is not automatically a negative example. The footer contains a geographical
string and its annotation depends on the agreed policy. Quarantine it if the
policy is unresolved. Initially prefer location-free invented footers.

## E. Morphological substitution can break fluency

Pattern:

```text
У <LOC_1> відкрили новий маршрут.
```

Reject an insertion that produces "У Черкаси ...". A valid gazetteer entry is not
necessarily a valid inflected surface form. Also review agreement with verbs,
adjectives, district names, and multiword locations.

## F. Unplanned locations are missing-label errors

The generator fills planned slots but also adds:

```text
Деталі опублікує адміністрація міста Львова.
```

"Львова" needs a reviewed annotation if geographical under the policy. Passing
placeholder checks does not make the post complete. Add the reviewed occurrence
or reject the candidate; never silently treat it as outside.

## G. Source labels are not automatically trustworthy

A source sentence may contain an unmarked country or city. Decorating it with
line breaks preserves the omission rather than fixing it. Audit the entire
parent and final post; store corrections only in the separate candidate record.

An original label can also become inappropriate when moved into a channel name,
quote, organization title, or footer. Context-preserving edits need review too.

## H. Similarity improvement without content improvement

Weak recipe:

```text
[emoji] <unrelated source sentence>

<another unrelated source sentence>

<identical footer repeated in every candidate>
```

Length and formatting may improve, but coherence and topic similarity may not.
A discriminator may also learn the repeated footer as a synthetic cue. Reject
incoherent assembly; diversify fit/evaluation families; require the controlled
AUC comparison as well as raw AUC.

## I. Forbidden shortcuts

- Rewriting a target post with different city names.
- Asking a generator to reproduce the supplied 26 examples' wording.
- Mining target-specific place names or channel signatures for insertion banks.
- Generating from source fold 0, then testing the extractor on fold 0.
- Sharing a parent across domain fit/evaluation through different paraphrases.
- Keeping only candidates that fool a discriminator fitted on evaluation posts.
- Treating all `test.csv` location placeholders as true empty labels.
- Accepting an agent's confident label assessment as human adjudication.
