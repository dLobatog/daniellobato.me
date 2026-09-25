# Transformers: Causal Teaching Pass

Scope: the seven concepts in `transformer-labs.js`, its family CSS and Node tests. The module contract, shared shell, Pretext integration, HTML pages and other families are unchanged. No libraries, theme changes or model services are added.

Every initial stage retains a concrete question, an immediately available computational action, visible legends and a real mechanism. Selected before/after values and a highlighted substituted equation stay outside the native calculation disclosure. Full tensors and settings remain optional. Comparisons use full-precision computations; displayed numbers are rounded to three decimals.

## tokenization

- Causal change: highlight the actual adjacent pieces before and after a merge, dim unchanged pieces, and reconcile weighted pair occurrences with the corpus token-count reduction. This is no longer just a changing segmentation display.
- Initial action: Apply next merge. One click applies one learned merge globally to the weighted training corpus and re-encodes the selected word with the updated frozen ranks.
- Equation linkage: `C(e,s) = 6 x 1 (newest) + 3 x 1 (widest) = 9`; the corpus shrinks from 79 to 70 tokens. The highlighted frequency terms refer to the highlighted pieces. The runner-up count explains selection and ties.
- Counterfactual quiz: if lowest becomes common after deployment without tokenizer retraining, does segmentation change? No; inference still applies the frozen ranks.
- Limitations: character-base, word-bounded BPE with a fixed tiny corpus and lexical tie-breaking. In this corpus pair occurrences do not overlap. In general overlapping occurrences need not equal the number of non-overlapping replacements. No arbitrary Unicode coverage claim.

## embeddings

- Causal change: retain the prior query as a dashed vector, show the before/after policy-minus-catalogue margin next to the plot, and explain why the same coordinate update affects the two documents differently. Nonselected vectors and candidates are dimmed but remain selectable.
- Initial action: Apply training step. Only q changes; candidate vectors stay fixed. The first dot-score margin changes from -0.500 to +0.074.
- Equation linkage: the selected coordinate displays `q_next[d] = q_before[d] - 0.2 * gradient[d]` with actual operands. Coordinate inspection is optional. The catalogue coefficient on dimension 0 is 3 versus 1 for the policy, explaining the margin effect.
- Counterfactual quiz: doubling only the catalogue vector doubles its dot score but leaves cosine unchanged.
- Limitations: hand-specified vectors and a supplied positive label, not measured semantic quality. q is optimized directly rather than through an encoder. Improved contrastive ranking need not mean reduced Euclidean distance to the positive. The training objective remains dot-product cross-entropy even if cosine ranking is inspected.

## positional

- Causal change: swapping literal sequence positions tracks the same key identity and shows its original-versus-current match score. Actual Q/K rotation arrows remain visible, alongside their token names, positions and radians.
- Initial action: Swap token order. The content vectors do not change; the selected relative displacement does. Switching positional schemes remains optional.
- Equation linkage: `q_base dot R(j-i) k_base / sqrt(2)` substitutes the current two vectors and highlights the rotated selected key. The no-position and additive modes instead substitute their actual Q/K vectors.
- Counterfactual quiz: shifting both rotary positions by five preserves the pair score because relative displacement is unchanged.
- Limitations: one rotary frequency, identity projections and unmasked attention to isolate position. No claim that pairwise shift invariance guarantees full-model long-context behavior. Additive encoding also changes V in this fixture.

## transformer-block

- Causal change: make the token intervention primary, keep a compact clickable residual architecture, and put the selected sublayer's actual mechanism and before/after output where the learner first looks. Attention shows cross-token paths; the MLP view shows separate token lanes. Other operations expose the selected row's transformation.
- Mobile inspection: the selected layer also reports its real before/after coordinate inside the architecture button, so a click has a visible consequence even when the full inspector is above the viewport. Both residual bypasses remain visible.
- Initial action: replace cache with buffer in the fixture sentence, keeping all other token rows fixed. Their explicit toy lookup vectors differ in one coordinate only: `[1,0,1]` versus `[0,0,1]`.
- Equation linkage: each selected stage exposes a substituted coordinate calculation: row normalization, Q projection, weighted V mixture, residual addition, second normalization, ReLU-MLP projection or final residual. The first attention output coordinate uses `W_O[:,0] = [1,0]`, so the displayed weighted V sum is exact. The changed source-token term is highlighted.
- Observed cause: at selected token keys, input, LN1 and Q are identical before/after. The first changed stage is attention; its coordinate 0 changes from 0.280 to -0.361. The MLP can subsequently respond to mixed context without directly reading token 0.
- Counterfactual quiz: without this block's attention branch, changing only token 0 cannot change token 3 through token-local LayerNorm, MLP or residual paths.
- Limitations: one fixed untrained head, width 3, hidden width 4, ReLU and LayerNorm epsilon 1e-5. The word-vector substitution is an explicit controlled fixture, not a learned semantic claim. Attention removal in the quiz assumes the later token's input is fixed; prior layers could already have mixed context.

## attention

- Causal change: replace query-stepping as the primary action with a value-only intervention. The selected V payload, output coordinate and weighted contribution change while the Q/K graph and weights remain exactly the same. The selected causal edge is emphasized; other edges are dimmed.
- Initial action: zero the value from cache while inspecting the query keys. This acts after projection and does not mutate X or any projection weights. Source/query nodes still support direct inspection, including masked future keys.
- Equation linkage: `o[0] = other_contributions + a_selected * V_selected[0]` shows all three substituted quantities and highlights the selected product. The first intervention removes exactly that summand.
- Counterfactual quiz: doubling one allowed V doubles that summand, not its attention weight and not necessarily the whole output.
- Limitations: one untrained head; artificial post-projection ablation isolates the distinction between matching and payload. No causal-attribution or factual-confidence claim. A masked source produces zero effect. The old optional prose-only next/previous trace was removed.

## kv-cache

- Causal change: appending selects the new slot automatically. Previous slots remain visible but dimmed; inspecting an old slot proves its selected key coordinate stayed identical. The before/after readout distinguishes a missing/new slot from an unchanged cached row.
- Initial action: Append next token. Exactly one K/V pair is newly projected and retained; existing pairs are not overwritten.
- Equation linkage: the selected token's key coordinate is expanded as `x @ W_K` with actual input/weight products. Storage is shown as previous bytes to current bytes: one extra pair adds `2 * 2 * 4 = 16` bytes.
- Counterfactual quiz: doubling KV head width while other dimensions and dtype stay fixed doubles retained storage, not necessarily latency.
- Limitations: one layer, one KV head, batch 1, FP32, fixed hidden inputs and predetermined decode tokens. Counts are storage/projection accounting, not benchmarks. The new query still scans the growing prefix; prefill/chunked decode retain causal masking.

## rag

- Causal change: highlight the exact manually audited supporting clause in the selected source, dim unrelated text, and show supporting source IDs before removal and after packing. Removed evidence is visibly marked absent rather than silently disappearing from the collection.
- Initial action: remove the supporting runbook from the prompt. Retrieval scores and the candidate claim stay fixed; support for the eviction rule is lost despite the keyword-heavy source A remaining.
- Equation linkage: `support = audited_source_ids intersect packed_source_ids`, with the audited IDs highlighted and the actual sets substituted. This is a fixture relation, not a probabilistic entailment equation.
- Counterfactual quiz: if B and C were both included and B is removed, the pinned-entry claim still has C, while the LRU claim loses B.
- Limitations: synthetic documents, term-count cosine and manually audited claim/source relations. No generator or automatic factuality verifier. Retrieval rank, citation presence and factual support are separate properties.
- Correctness follow-up: source captions and the runbook action now depend on the selected claim. Inspecting B or C for an unsupported claim no longer says a highlighted supporting clause is available; the latency question does not imply that any supporting source exists.

## Verification

Owned Node tests: 33 passing. Coverage includes the original arithmetic plus BPE count reconciliation and frozen inference ranks, exact gradient-driven margin changes, value-only ablation for every query/key pair, the first cross-token effect in the complete block, cache append immutability, verbatim evidence spans, claim-specific captions, equation visibility and a nontrivial numerical/evidence outcome for every initial action. Regression checks also cover all rotary arrow endpoints, nearby vector legends, substituted token labels, the selected architecture value and readable inactive-label CSS.

## Render Review

- Parent captures reviewed at 1440 and 390 pixels: initial and after-action images for all seven concepts. Bottom captures reviewed at both widths for tokenization, embeddings, positional, transformer-block and RAG; attention bottom was reviewed at 1440. The parent directory did not contain KV-cache bottom captures at either width or an attention bottom capture at 390 when checked; these were not claimed as seen.
- Screenshot-driven fixes in owned JS/CSS: put the embedding ID-to-document legend with the vectors, reduce unused RoPE plot space while retaining every endpoint and the radians/line-style legend, show selected-layer values inside the residual architecture, and replace whole-node opacity on necessary labels/evidence with theme-aware muted text. Only shapes and arrows retain opacity de-emphasis.
- Fresh captures and reports are under `/tmp/atelier-causal-transformers-final`, not the parent's directory. An isolated headless process used the existing local server; it did not access or reload the shared in-app browser. The light-theme run passed 56 before/after width checks across 320, 390, 720 and 1440 pixels with zero errors. A separate dark-theme mobile run passed 14 checks with zero errors.
- Fresh visual review: all seven initial stages at 1440 and 390; embedding, positional and RAG after-action states at both widths; transformer attention and MLP inspectors plus both residual rails at both widths; unsupported RAG states at both widths. All seven dark-theme mobile initial stages were also visually reviewed. Width checks at 320 and 720 are automated checks, not claimed visual screenshot review.
- The parent's captures predate the compact positional view and text-contrast changes. The fresh captures use the updated family CSS. In the fresh 390px captures the embedding and positional equations end at approximately 828px and 844px respectively, within the 900px capture, with no overlapping labels.

Only the family JS, family CSS, corresponding test file and this document are repository edits. No shared files, libraries, commits or pushes.
