# Neural-network chapter validation

Date: 2026-09-24. Scope: nine rebuilt lessons in `atelier/neural-network-basics.html`, plus shared reading-theme and viewer changes. The other chapters have not received the same concept-level rebuild.

## Rendered review

- Personally inspected actual in-app browser screenshots of all nine expanded lessons at 1440 x 900, 720 x 900, and 390 x 844.
- Checked all nine expanded layouts at 320 x 844 for document, dialog, and lesson horizontal overflow; none. Also inspected screenshots of the chain-rule numerical probe and learning-rate schedule at this width.
- Inspected the final backpropagation update, independent sigmoid versus competing softmax changes, activation regimes, divergent optimization, the schedule strip, and the optional dark reading theme.
- All 17 public pages load with headings, content, and the shared theme toggle at 390px, without page-level horizontal overflow. This is a page-load smoke test, not a visual certification of every legacy visualization.
- Browser console error check was empty during these checks.

## Interaction and content checks

- Stage-background click opens the viewer. Close, outside click, and Escape close it. Tab focus remains within the dialog and returns to the opener. Reopening resets the viewer scroll to the top.
- Next/Previous and inspectors update real values, preserve state between inline and expanded views, and support keyboard interaction.
- Delivery example: prediction 15 minutes, observed 19 minutes, half-squared loss 8; gradients (-8, -4, -4); one simultaneous 0.1 SGD update gives prediction 17.4 and loss 1.28.
- Chain-rule numerical probe distinguishes the local prediction (-0.08) from the actual finite loss change (-0.0798).
- Increasing the dog's logit leaves the cat's independent sigmoid probability unchanged, but lowers its softmax share.
- All nine formula disclosures opened at mobile width without KaTeX errors or annotation overflow. Correct quiz choices produce correct feedback.
- Formula explanations, code, questions, and visuals use the same computed examples. Adam includes bias correction; the optimizer comparison does not claim a universal winner.

## Regression suite

`node --test atelier/tests/*.test.cjs`: 49 tests, 47 pass, 0 fail, 2 optional browser tests skipped in this invocation. Browser checks above were performed separately against the real running site.

Coverage includes finite-difference gradients, stable sigmoid/softmax, ReLU's undefined derivative at zero versus its implementation convention, exact SGD contractions, an independent Adam recurrence, schedule boundaries, state bounds, module registration, optional theme persistence, local asset references, and Pretext stale-text/font regressions.

## Limits and follow-up

- One linear neuron intentionally isolates the learning loop; it is not a substitute for tracing a multilayer nonlinear network next.
- The deterministic quadratic teaches update mechanics, not generalization or a production optimizer ranking.
- The broader library retains legacy visualization layouts and some duplicated controls. Review them individually, rather than treating the lighter shell as completion.
- Comprehension still needs learner review: can someone predict the changed output or gradient before revealing it?
