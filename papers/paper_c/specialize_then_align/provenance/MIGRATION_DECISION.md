# Migration decision

Paper C v2 is scientifically incompatible with the predecessor's binary,
prompt-only reference-centering study. It therefore uses a new package, study ID,
configuration schema, lock chain, artefact namespace, and manuscript.

The predecessor is neither a parent nor a superseded protocol. Its files are not
imported at runtime. The only relationship is an auditable historical pointer in
`LEGACY_REFERENCE_CENTERING.json`.

Reasons for a clean boundary:

- the unit changed from one-token binary prompt classification to a structured
  three-action event;
- the intervention changed from label-derived DPO to adjudicated cross-model
  structured preferences;
- mortgage adds dated policy, jurisdiction, context, and human-review contracts;
- the predecessor smoke failed its policy/reference identity check before Stage
  2, so it supplies no reusable result;
- its original six-manifest data snapshot contains no mortgage examples.

No new study result may cite the predecessor smoke as evidence. Generic hashing
and path-safety patterns were independently reimplemented and tested here.
