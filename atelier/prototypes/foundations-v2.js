(function () {
  'use strict';

  const studio = window.AtelierStudio;
  if (!studio?.chapters?.foundations) return;

  const baseChapter = studio.chapters.foundations;
  const baseById = Object.fromEntries(baseChapter.sections.map((section) => [section.id, section]));

  const candidateSections = [
    {
      id: 'distribution',
      label: 'Concept 01',
      nav: 'Distribution',
      title: 'A distribution is just probability budget spread over outcomes',
      intro:
        'If one outcome gets most of the budget, it should show up most of the time in repeated samples.',
      intuition:
        'Probability is mass you spend across possibilities. A distribution is that spending plan; sampling is what the plan looks like when the world starts drawing from it.',
      prompt:
        'Switch between the balanced and skewed presets. The one thing to notice is that the sample histogram gradually takes on the same shape as the probability bars above it.',
      remember:
        'A distribution is probability mass over outcomes, and samples follow that shape.',
      mathTitle: 'Discrete and continuous',
      mathLines: [
        '\\sum_i p(x_i) = 1',
        '\\int_{-\\infty}^{\\infty} p(x)\\, dx = 1',
      ],
      mathHelp: [
        'For discrete outcomes, all the weights must add up to 1.',
        'For continuous outcomes, probability is area under the curve, not curve height by itself.',
      ],
      notes: baseById.distribution.details,
      quiz: baseById.distribution.quiz,
      viz: baseById.distribution.viz,
      controls: baseById.distribution.controls,
      presets: baseById.distribution.presets,
    },
    {
      id: 'expectation',
      label: 'Concept 02',
      nav: 'Expectation',
      title: 'Expectation is the balance point of the whole distribution',
      intro:
        'The average is not just a familiar statistic. It is the place where all the probability mass would balance if you hung it on a line.',
      intuition:
        'Treat each outcome like a weight at a position. The expectation is the single point where the weighted system balances. That is why long-run averages drift toward it.',
      prompt:
        'Shift mass to the right or left and watch the balance point follow. The key lesson is that expectation moves with where the weight really lives, not with the most likely single outcome alone.',
      remember:
        'Expectation is where the distribution balances, and repeated sampling drifts toward that point.',
      mathTitle: 'Expectation and variance',
      mathLines: [
        'E[X] = \\sum_i x_i\\,p(x_i)',
        'E[X] = \\int x\\,p(x)\\,dx',
        '\\operatorname{Var}(X) = E[(X - E[X])^2]',
      ],
      mathHelp: [
        'Each outcome is weighted by how often it occurs.',
        'Variance is just the average squared distance from that balance point.',
      ],
      notes: baseById.expectation.details,
      quiz: baseById.expectation.quiz,
      viz: baseById.expectation.viz,
      controls: baseById.expectation.controls,
      presets: baseById.expectation.presets,
    },
    {
      id: 'bayes',
      label: 'Concept 03',
      nav: 'Bayes',
      title: 'Forget the formula. Picture 1,000 people and just count.',
      intro:
        'A positive test feels like proof. But to see what it really means, stop thinking about abstract probabilities and start thinking about a concrete group of people.',
      intuition:
        'Picture 1,000 people. Some fraction actually has the condition — that is the prior. Now run the test on everyone. A few sick people test positive (true positives). But a much larger healthy crowd also produces some positives (false positives), because no test is perfect and the healthy crowd is huge. The posterior is just a head-count: of everyone who tested positive, what fraction actually has it? When the condition is rare, the false-positive crowd can easily outnumber the true positives — and that is why a good test can still give a surprisingly low posterior.',
      prompt:
        'Set the prior to 1 % and watch: out of 1,000 people only ~10 are sick, so even a 95 %-sensitive test catches roughly 9. Meanwhile ~950 healthy people at 8 % false-positive rate produce ~76 false alarms. 9 true positives drowned in 76 false positives: the posterior is about 10 %. Now raise the prior to 35 % and see the true-positive bar overwhelm the false-positive bar.',
      remember:
        'New evidence does not determine your beliefs in a vacuum — it updates prior beliefs. The posterior is just: true positives divided by everyone who tested positive.',
      mathTitle: 'Posterior formula',
      mathLines: ['P(H \\mid E) = \\frac{P(E \\mid H)\\,P(H)}{P(E)}'],
      mathHelp: [
        'P(H) is the prior — what fraction of the 1,000 actually has the condition before any test.',
        'P(E|H) is the likelihood — of those who have it, what fraction tests positive.',
        'P(E) is the total positive rate — true positives plus false positives. This is the denominator that makes the formula just a proportion.',
      ],
      notes: baseById.bayes.details,
      geometry: baseById.bayes.geometry,
      quiz: baseById.bayes.quiz,
      viz: baseById.bayes.viz,
      controls: baseById.bayes.controls,
      presets: baseById.bayes.presets,
    },
    {
      id: 'entropy',
      label: 'Concept 04',
      nav: 'Entropy',
      title: 'Entropy is average surprise; cross-entropy adds the cost of being wrong',
      intro:
        'Some uncertainty belongs to the task itself. The rest is the penalty your model pays for betting on the wrong distribution.',
      intuition:
        'Entropy asks: on average, how surprised am I by the true world? Cross-entropy asks: how surprised am I when I use my model instead? The gap between them is pure mismatch.',
      prompt:
        'First move p toward 50/50 and watch intrinsic uncertainty rise. Then keep p fixed and move q away from it; the extra gold mismatch cost grows even though the world itself did not change.',
      remember:
        'Entropy belongs to the world. Cross-entropy is world uncertainty plus the price of the model being off.',
      mathTitle: 'Entropy and cross-entropy',
      mathLines: [
        'H(p, q) = H(p) + KL(p \\parallel q)',
        '\\mathcal{L}_{CE} = -\\sum_x p(x)\\log q(x)',
      ],
      mathHelp: [
        'H(p): uncertainty already present in the real distribution.',
        'KL(p || q): extra cost caused purely by the model believing the wrong probabilities.',
      ],
      notes: baseById.entropy.details,
      quiz: baseById.entropy.quiz,
      viz: baseById.entropy.viz,
      controls: baseById.entropy.controls,
      presets: baseById.entropy.presets,
    },
    {
      id: 'loss',
      label: 'Concept 05',
      nav: 'Loss',
      title: 'A loss function is just a scoring rule for mistakes',
      intro:
        'Training only knows how to improve a number. The loss is the number you choose to represent the kind of mistake you care about.',
      intuition:
        'Take the same prediction error and ask how different scoring rules punish it. Squared loss hates large misses, absolute loss is more tolerant to outliers, and log-loss punishes confident wrongness brutally.',
      prompt:
        'Compare how the same miss is scored under different losses. The lesson is not the formula names; it is what kind of behavior each loss rewards or punishes.',
      remember:
        'Choosing a loss means choosing what kind of wrongness matters most.',
      mathTitle: 'Common losses',
      mathLines: [
        '\\mathcal{L}_{MSE} = \\tfrac{1}{n}\\sum_i (y_i - \\hat{y}_i)^2',
        '\\mathcal{L}_{MAE} = \\tfrac{1}{n}\\sum_i |y_i - \\hat{y}_i|',
        '\\mathcal{L}_{\\log} = -\\tfrac{1}{n}\\sum_i\\bigl[y_i\\log\\hat{p}_i + (1-y_i)\\log(1-\\hat{p}_i)\\bigr]',
      ],
      mathHelp: [
        'MSE amplifies big mistakes because errors are squared.',
        'MAE treats every unit of error linearly.',
        'Log-loss explodes when the model is confidently wrong about a probability.',
      ],
      notes: baseById.loss.details,
      quiz: baseById.loss.quiz,
      viz: baseById.loss.viz,
      controls: baseById.loss.controls,
      presets: baseById.loss.presets,
    },
  ];

  function renderMathLines(lines) {
    return lines
      .map(
        (line) => `
          <div class="formula-line">
            <span class="math-render" data-math="${line.replaceAll('"', '&quot;')}">${line}</span>
          </div>
        `
      )
      .join('');
  }

  function renderQuiz(section) {
    if (!section.quiz) return '';
    return `
      <details class="fold study-tool">
        <summary>Quick check</summary>
        <div class="fold-body">
          <div class="quiz-card" data-quiz="${section.id}">
            <p class="tool-kicker">One check</p>
            <p class="quiz-prompt">${section.quiz.prompt}</p>
            <div class="quiz-options">
              ${section.quiz.options
                .map(
                  (option, index) => `
                    <button
                      type="button"
                      class="quiz-option"
                      data-correct="${option.correct ? 'true' : 'false'}"
                      data-explanation="${option.explanation.replaceAll('"', '&quot;')}"
                    >
                      ${index + 1}. ${option.text}
                    </button>
                  `
                )
                .join('')}
            </div>
            <p class="quiz-feedback" data-feedback>Choose an answer to check your understanding.</p>
          </div>
        </div>
      </details>
    `;
  }

  function renderMath(section) {
    return `
      <details class="fold study-tool">
        <summary>Formula</summary>
        <div class="fold-body">
          <div class="formula-card">
            <p class="tool-kicker">${section.mathTitle}</p>
            ${renderMathLines(section.mathLines)}
            <div class="prototype-formula-help">
              ${section.mathHelp.map((item) => `<p>${item}</p>`).join('')}
            </div>
          </div>
        </div>
      </details>
    `;
  }

  function renderNotes(section) {
    return `
      <details class="fold study-tool">
        <summary>Longer notes</summary>
        <div class="fold-body">
          <div class="formula-card">
            <p class="tool-kicker">If you want the slower version</p>
            <ul>
              ${section.notes.map((item) => `<li>${item}</li>`).join('')}
            </ul>
          </div>
        </div>
      </details>
    `;
  }

  function renderGeometry(section) {
    if (!section.geometry) return '';
    const g = section.geometry;
    return `
      <details class="fold study-tool">
        <summary>Geometric view</summary>
        <div class="fold-body">
          <div class="formula-card geometry-card">
            <p class="tool-kicker">${g.title}</p>
            <p class="tool-note">${g.body}</p>
            <div class="bayes-geometry" data-bayes-geometry>
              <div class="bayes-geometry-visual" data-bayes-visual></div>
              <div class="bayes-measure-grid" data-bayes-measures></div>
              <p class="tool-note bayes-geometry-note" data-bayes-note></p>
            </div>
            ${renderMathLines([g.formula])}
            <p class="tool-note">${g.note}</p>
          </div>
        </div>
      </details>
    `;
  }

  function renderConceptPager(previousSection, nextSection) {
    if (!previousSection && !nextSection) return '';
    return `
      <nav class="concept-pager" aria-label="Concept navigation">
        ${
          previousSection
            ? `<a class="concept-pager-link" href="#${previousSection.id}"><span class="concept-pager-label">Previous concept</span><span class="concept-pager-title">${previousSection.nav}</span></a>`
            : '<span class="concept-pager-spacer" aria-hidden="true"></span>'
        }
        ${
          nextSection
            ? `<a class="concept-pager-link concept-pager-link-next" href="#${nextSection.id}"><span class="concept-pager-label">Next concept</span><span class="concept-pager-title">${nextSection.nav}</span></a>`
            : '<span class="concept-pager-spacer" aria-hidden="true"></span>'
        }
      </nav>
    `;
  }

  function renderConcept(section, index) {
    const article = document.createElement('article');
    article.className = 'concept-card candidate-concept-card';
    article.id = section.id;
    article.innerHTML = `
      <div class="candidate-head concept-copy">
        <p class="concept-label">${section.label}</p>
        <h2>${section.title}</h2>
        <p class="candidate-intro">${section.intro}</p>
      </div>

      <div class="candidate-layout">
        <div class="candidate-teaching">
          <article class="fact-card">
            <p class="fact-label">Core intuition</p>
            <p>${section.intuition}</p>
          </article>
          <article class="fact-card">
            <p class="fact-label">Try this</p>
            <p>${section.prompt}</p>
          </article>
          <article class="fact-card candidate-remember">
            <p class="fact-label">Say it back</p>
            <p>${section.remember}</p>
          </article>
        </div>

        <section class="viz-panel" data-viz="${section.viz}">
          <div class="viz-toolbar">
            <span class="viz-toolbar-label">Interactive visualization</span>
            <button type="button" class="viz-expand-button" aria-expanded="false">Expand view</button>
          </div>
          <div class="viz-shell">
            <div class="viz-stage" tabindex="0" role="button" aria-label="Expand visualization">
              <div class="viz-stage-root" data-stage-root></div>
            </div>
            <div class="lab-readout">
              <strong class="takeaway-label">Current read</strong>
              <div class="takeaway-line" data-role="takeaway"></div>
              <div class="metric-row" data-role="metrics"></div>
            </div>
            <div class="viz-support">
              <div class="preset-row">
                ${section.presets
                  .map(
                    (preset, presetIndex) =>
                      `<button class="preset-button${presetIndex === 1 || (presetIndex === 0 && section.presets.length === 1) ? ' active' : ''}" type="button" data-preset='${JSON.stringify(
                        preset.values
                      )}'>${preset.label}</button>`
                  )
                  .join('')}
              </div>
              <div class="control-row">
                ${section.controls
                  .map(
                    (control) => `
                      <div class="control-group">
                        <label for="${section.id}-${control.key}">
                          <span>${control.label}</span>
                          <span data-value="${control.key}"></span>
                        </label>
                        <input
                          id="${section.id}-${control.key}"
                          type="range"
                          min="${control.min}"
                          max="${control.max}"
                          step="${control.step}"
                          value="${control.value}"
                          data-key="${control.key}"
                          data-format="${control.format}"
                        >
                      </div>
                    `
                  )
                  .join('')}
              </div>
            </div>
          </div>
        </section>
      </div>

      <div class="study-tool-grid candidate-tools">
        ${renderMath(section)}
        ${renderGeometry(section)}
        ${renderQuiz(section)}
        ${renderNotes(section)}
      </div>
      ${renderConceptPager(candidateSections[index - 1] || null, candidateSections[index + 1] || null)}
    `;

    studio.mountVisualization(article, section);
    studio.mountQuiz(article);
    studio.mountGeometry(article, section);
    if (section.id === 'bayes') installBayesReadoutRewrite(article);
    return article;
  }

  function installBayesReadoutRewrite(article) {
    const readout = article.querySelector('.lab-readout');
    const metricRow = article.querySelector('.metric-row');
    if (!readout || !metricRow) return;

    let chain = readout.querySelector('.prototype-bayes-chain');
    if (!chain) {
      chain = document.createElement('p');
      chain.className = 'prototype-bayes-chain';
      metricRow.insertAdjacentElement('afterend', chain);
    }

    const rewrite = () => {
      const cards = Array.from(metricRow.querySelectorAll('.metric-card'));
      const byLabel = new Map();
      cards.forEach((card) => {
        const label = card.querySelector('.metric-label')?.textContent?.trim() || '';
        const value = card.querySelector('.metric-value')?.textContent?.trim() || '';
        byLabel.set(label, { card, value });
      });

      cards.forEach((card, index) => {
        const labelNode = card.querySelector('.metric-label');
        if (!labelNode) return;
        const raw = labelNode.textContent.trim();
        if (raw === 'P(sick ∩ +)') labelNode.textContent = 'True positives in the whole population';
        else if (raw === 'P(healthy ∩ +)') labelNode.textContent = 'False positives in the whole population';
        else if (raw === 'Σ' || raw === 'P(+)') labelNode.textContent = 'All positive tests together';
        else if (raw === 'P(sick | +)' || raw === 'P(H|+)' || raw === 'Posterior P(H|+)') {
          labelNode.textContent = 'Share of positives that are truly sick';
        } else if (raw === 'True positives') {
          labelNode.textContent = 'True positives inside the positive-test pool';
        } else if (raw === 'False positives') {
          labelNode.textContent = 'False positives inside the positive-test pool';
        } else if (raw === 'Posterior') {
          labelNode.textContent = 'Share of positives that are truly sick';
        } else if (!raw && cards.length === 4) {
          const fallback = [
            'True positives in the whole population',
            'False positives in the whole population',
            'All positive tests together',
            'Share of positives that are truly sick',
          ][index];
          if (fallback) labelNode.textContent = fallback;
        }
      });

      const truePositive =
        byLabel.get('P(sick ∩ +)')?.value ||
        byLabel.get('True positives')?.value ||
        cards[0]?.querySelector('.metric-value')?.textContent?.trim() ||
        '';
      const falsePositive =
        byLabel.get('P(healthy ∩ +)')?.value ||
        byLabel.get('False positives')?.value ||
        cards[1]?.querySelector('.metric-value')?.textContent?.trim() ||
        '';
      const positivePool =
        byLabel.get('Σ')?.value ||
        byLabel.get('P(+)')?.value ||
        cards[2]?.querySelector('.metric-value')?.textContent?.trim() ||
        '';
      const posterior =
        byLabel.get('P(sick | +)')?.value ||
        byLabel.get('P(H|+)')?.value ||
        byLabel.get('Posterior P(H|+)')?.value ||
        byLabel.get('Posterior')?.value ||
        cards[3]?.querySelector('.metric-value')?.textContent?.trim() ||
        cards[0]?.querySelector('.metric-value')?.textContent?.trim() ||
        '';

      if (truePositive && falsePositive && positivePool && posterior) {
        chain.textContent =
          `Chain: ${truePositive} true-positive mass + ${falsePositive} false-positive mass = ${positivePool} total positive tests. ` +
          `The posterior is the true-positive share inside that pool: ${posterior}.`;
      } else {
        chain.textContent =
          'Chain: true positives plus false positives make the whole positive-test pool. The posterior is the true-positive share inside that pool.';
      }
    };

    rewrite();
    const observer = new MutationObserver(rewrite);
    observer.observe(metricRow, { childList: true, subtree: true, characterData: true });
  }

  function mount() {
    const root = document.getElementById('candidate-root');
    const map = document.getElementById('candidate-map');
    if (!root || !map) return;

    map.innerHTML = candidateSections.map((section) => `<a href="#${section.id}">${section.nav}</a>`).join('');
    root.innerHTML = '';
    candidateSections.forEach((section, index) => {
      root.appendChild(renderConcept(section, index));
    });
    studio.renderMathBlocks(root);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  } else {
    mount();
  }
})();
