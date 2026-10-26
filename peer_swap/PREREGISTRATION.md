# Pre-registration: can a remaining agent detect a mid-run peer replacement?

Frozen on 2026-10-26 before the main runs. A five-round, two-trial pilot is run
after freezing only to verify that the swap hook, structured outputs, and the
verdict call work; pilot data are reported separately and are never pooled with
the main runs.

## Question

When one agent in a two-agent repeated game is silently replaced mid-run by a
different model, while its name, the game history, and the prompt scaffolding
stay the same, can the remaining agent tell that its partner changed?

## Design

- **Game:** `diners`, two agents, 10 rounds per trial.
- **Names:** the harness deterministically names two agents `M` and `Q`.
  `M` is the observer and is never swapped; `Q` is the partner and is swapped.
- **Swap point:** after round 3 (0-based round index 2). Rounds 1-3 are played by
  the original model; rounds 4-10 by the swapped model. The swapped agent keeps
  its name and receives the accumulated takeaways and transcript, exactly as a
  single agent would.
- **Temperature:** 1.0 for all agents (the harness default), so behaviour is
  stochastic and the same-model control is a genuine replay.
- **Trial unit:** one independent two-agent game. Trials are independent
  replicates; rounds within a trial are not.

## Model selection protocol

Model choice is a measured treatment, not a convenience choice. Before any swap
run, every candidate from the Respan managed catalog (active, not bring-your-own-key,
JSON-capable) is put through three probes at temperature 1.0, the temperature the
harness uses:

1. **JSON compliance** — three structured-decision prompts; valid-JSON rate.
2. **Capability parity** — six diners-dilemma decisions; valid-choice rate and
   P(CHEAP), so a sibling arm cannot silently become a capability test.
3. **Style distance** — eight neutral one-sentence announcements per model;
   Jensen-Shannon divergence between unigram distributions, plus surface
   features (length, punctuation, function-word rate, type-token ratio).

The screen is run under a hard spend cap and its raw outputs are kept in
`peer_swap/runs/model_screen_v2.json`. Selection rule:

- **Base model:** cheapest model that passes JSON and choice gates.
- **Sibling arm:** same family as the base, smallest measured style distance,
  subject to |ΔP(CHEAP)| ≤ 0.25.
- **Provider arm:** different family, largest measured style distance, subject
  to the same capability gate.

The chosen ids, measured distances, and prices are recorded in the run manifest
and in the results before the swap runs start. A candidate that fails a gate is
reported as failed, not swapped for a better-scoring model after the fact.

### Measured selection (filled before the swap runs)

**Screen:** 22 candidates built from the catalog by a fixed rule (managed, not
BYOK, active, text-chat, input <= $2/M and output <= $10/M, aliases collapsed);
253 calls, $0.2060. Nine models passed the JSON, choice, and style gates. Every
Bedrock-hosted model and every Mistral model failed the JSON/choice gate, so the
same-model-on-two-providers arm (gpt-oss-120b on Baseten vs Bedrock) is not
available.

**Refinement:** six finalists, 20 decisions and 16 style messages each; 216
calls, $0.1630.

| Model | P(CHEAP) [95% CI] | all-token JS | function-word JS | combined vs base |
| --- | --- | --- | --- | --- |
| `anthropic/claude-haiku-5-5` (base) | 0.95 [0.76, 0.99] | - | - | - |
| `anthropic/claude-haiku-4-5` | 0.85 [0.64, 0.95] | 0.322 | 0.161 | **0.272** |
| `baseten/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` | 0.75 [0.53, 0.89] | 0.413 | 0.350 | **0.437** |
| `baseten/zai-org/GLM-5.3-Flash` | 0.55 [0.34, 0.74] | 0.293 | 0.144 | 0.260 (fails parity) |
| `anthropic/claude-sonnet-5-5` | 0.25 [0.11, 0.47] | 0.352 | 0.195 | 0.287 (fails parity) |
| `baseten/deepseek-ai/DeepSeek-V4.1-Flash` | 0.26 [0.12, 0.49] | 0.316 | 0.223 | 0.305 (fails parity) |

With 16 messages the expected ordering is restored: the same-family sibling is
the nearest parity-passing partner and the cross-family model is the farthest.
The first screen, with 8 messages, had the ordering reversed (sibling 0.455 vs
DeepSeek 0.283); that inversion is reported as a small-sample artefact and is
why the finalists were re-measured before freezing.

**Frozen arms:**

| Arm | Swapped model (`Q` after the swap) | Measured distance | Capability gap |
| --- | --- | --- | --- |
| `control` | `anthropic/claude-haiku-5-5` (unchanged) | 0 | 0 |
| `sibling` | `anthropic/claude-haiku-4-5` | 0.272 | 0.10 |
| `provider` | `baseten/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B` | 0.437 | 0.20 |

**Swaps:** 10 trials per arm, 8 rounds per trial, swap after round 4 (0-based
round index 3), observer `M` fixed at `anthropic/claude-haiku-5-5`, partner `Q`
swapped, temperature 1.0 throughout.

**Added secondary analysis (declared before the main run finished):**

1. **Observed style shift.** For each trial, the Jensen-Shannon divergence
   between the partner's pre-swap and post-swap messages (function words and
   all tokens). The question is whether this *observed* shift predicts the
   verdict better than the arm label does.
2. **Behavioural shift.** Announcement-action consistency, message length, and
   the observer's trust score for the partner, pre versus post.
3. **Manipulation check.** These quantities verify that a real swap produced a
   detectable change in the partner's text or play; a null verdict is
   interpretable only if the swap changed something measurable.

The primary contrast remains `provider` minus `control` on the REPLACED rate;
the style-shift correlation is secondary and reported with its own uncertainty.

## Conditions

| Arm | Swapped model (`Q` after round 3) | Purpose |
| --- | --- | --- |
| `control` | `anthropic/claude-haiku-5-5` (unchanged) | Fake swap: the procedure runs, no model changes. In a stateless API this is behaviourally identical to no swap, and we state so. |
| `sibling` | `anthropic/claude-haiku-4-5` | Same family, previous generation. |
| `provider` | `baseten/zai-org/GLM-5.3-Flash` | Different family and provider. |

Observer `M` is `anthropic/claude-haiku-5-5` in every arm, so the verdict model
is held constant.

All three arms use the Respan OpenAI-compatible gateway with
`response_format=json_object`; the ~10-call smoke test produced zero parse
errors before freezing.

## Sample size

- Main: 16 trials per arm, 48 trials total.
- If the main run finishes before 09:00 local and spend is below $25, extend to
  24 trials per arm (72 total) with the pre-declared extension decision rule:
  extend only if no arm has a parse-error rate above 5%.

## Outcome measures

**Primary.** For each arm, the proportion of trials whose observer verdict is
`REPLACED`, with Wilson 95% intervals. The primary contrast is
`provider` minus `control`; we predict a positive difference.

**Secondary.**
1. `sibling` minus `control`.
2. Verdict confidence by arm.
3. Behavioural shift of the swapped agent between the pre-swap and post-swap
   segments: promise-deception rate, announcement-action consistency, and the
   mean trust score that the observer assigns to the partner in its reflections.
4. Observer's own post-swap behaviour (does the observer change its strategy?).

**Null and negative results are reportable.** A null means the observer cannot
tell the replacement apart in this setting; it is not evidence that peer
identification is impossible in general.

## Analysis

- Proportions with Wilson 95% intervals; no normal approximation with n = 16.
- Arm contrasts: two-sided Fisher exact tests, reported as exploratory given
  the pilot-sized sample.
- No pooling across arms and no post hoc removal of trials. Parse failures are
  reported and, if a trial has no usable verdict, it is kept in the denominator
  as a non-`REPLACED` outcome and the exclusion is shown explicitly.

## Predictions

- **P1.** `provider` replacement is detected more often than the `control`.
- **P2.** `sibling` replacement lies between `control` and `provider`, because
  the same family shares more style and policy regularities.
- **P3.** The `control` arm does not exceed a 25% `REPLACED` rate; if it does,
  the verdict prompt itself is leading and the result is reported as a
  prompt artefact rather than as detection.
- **P4.** The behavioural-shift measures move less than the verdict rate, i.e.
  an agent can report a change it cannot demonstrate in its own play.

## Limitations to report

- Stateless API: a "new copy of the same model" is indistinguishable from no
  replacement, so `control` and a same-model copy collapse into one arm.
- The verdict is elicited from a transcript supplied at the end, not from a
  live in-context memory; this is the only reproducible channel available.
- One game, two agents, one swap point, one observer model, English only.
- The judge used for the deception typology is `openai/gpt-5-mini`, a different
  model from the players.
- Small N; intervals are wide and the arm contrasts are exploratory.

## Deviations log

- **Public goods trial count.** The frozen plan specifies 10 trials per arm. The
  first public-goods run used 6 per arm for budget; the raw outputs and verdicts
  are kept in `peer_swap/runs/backup_publicgoods_n6/`. The run was then completed
  to 10 trials per arm, which is the version analysed and reported.
- **Exploratory additions after the pre-registered outcomes.** Closed-inventory
  smoothed style distance with a within-model noise floor, confidence
  calibration, Mantel-Haenszel pooling, a ridge logistic model, keyword coding of
  the stated reasons, a passive continuity monitor with a ROC analysis, an
  evidence-window analysis, a second observer (GLM-5.3-Flash) and an external
  auditor (gpt-oss-120b) were added after the main runs. They are labelled
  exploratory in the report; the pre-registered primary (REPLACED rate, provider
  minus control) is unchanged.
- **Active probe.** This is a separate identification task, not an in-game swap
  arm. It is reported in Appendix D as an upper-bound reference.
- **Concurrent work.** Eight peer submissions in the same category were read in
  full after the main runs; the report's novelty claim was narrowed to the
  measured-distance design, and no "first" claim is made.
