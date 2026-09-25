/* Small, self-contained lessons. Inline and focus views share one state. */
(() => {
  const instances = new Map();
  const count = n => Math.round(n).toLocaleString('en-US');
  const percent = n => `${Number((100 * n).toFixed(1))}%`;
  const bits = n => n.toFixed(3);
  const entropy = p => -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
  const crossEntropy = (p, q) => -p * Math.log2(q) - (1 - p) * Math.log2(1 - q);
  const dieShapes = { fair: [1,1,1,1,1,1], high: [1,1,1,1,2,4], extremes: [1,0,0,0,0,1] };
  function expectation(weights) {
    const total = weights.reduce((a,b) => a+b, 0);
    return weights.reduce((sum, weight, i) => sum + (i+1) * weight / total, 0);
  }
  function bayes(prior) {
    const sick = 10000 * prior;
    const healthy = 10000 - sick;
    const truePositive = sick * 0.99;
    const falsePositive = healthy * 0.01;
    return { sick, healthy, truePositive, falsePositive, positive: truePositive + falsePositive, posterior: truePositive / (truePositive + falsePositive) };
  }
  const choice = (action, value, text, selected) => `<button type="button" class="lesson-choice" data-action="${action}" data-value="${value}" aria-pressed="${selected}">${text}</button>`;

  function bayesMarkup(s) {
    const b = bayes(s.prior);
    const titles = ['Start with everyone.', 'The same test, two kinds of positive.', 'A positive result changes the denominator.'];
    const messages = [
      `Of 10,000 people, <strong>${count(b.sick)} are sick</strong>. The other ${count(b.healthy)} are healthy. This starting fraction, ${percent(s.prior)}, is the <em>prior</em>.`,
      `The test finds 99% of sick people and falsely flags 1% of healthy people. That produces <strong>${count(b.truePositive)} real detections and ${count(b.falsePositive)} false alarms</strong>. Both look like a positive test.`,
      `You know the result is positive, so set aside everyone who tested negative. Of the <strong>${count(b.positive)} positives</strong> left, <strong>${count(b.truePositive)} are sick</strong>. That fraction is the <em>posterior</em>.`,
    ];
    const share = s.step === 2 ? b.posterior : s.prior;
    return `<div class="lesson lesson-bayes">
      <div class="lesson-options"><span class="lesson-caption">How common is the disease?</span><div class="lesson-choices" role="group" aria-label="Disease prevalence">
        ${[.01, .1, .5].map(p => choice('prior', p, `${percent(p)} of people`, p === s.prior)).join('')}
      </div></div>
      <nav class="lesson-steps" aria-label="Bayes walkthrough">
        ${['Population', 'Test results', 'Only positives'].map((name, i) => `<button type="button" data-action="step" data-value="${i}" ${i === s.step ? 'aria-current="step"' : ''}><span>${i + 1}</span>${name}</button>`).join('')}
      </nav>
      <div class="lesson-scene" aria-live="polite" aria-atomic="true">
        <h3>${titles[s.step]}</h3>
        <p class="lesson-narrative">${messages[s.step]}</p>
        ${s.step === 1 ? `<div class="test-branches">
          <button type="button" class="test-branch true-branch" data-action="inspect" data-value="sick" aria-pressed="${s.inspect === 'sick'}"><span>${count(b.sick)} sick people</span><span class="branch-arrow">99% test positive</span><strong>${count(b.truePositive)}</strong><span>real detections</span><small>${count(b.sick - b.truePositive)} test negative</small></button>
          <button type="button" class="test-branch false-branch" data-action="inspect" data-value="healthy" aria-pressed="${s.inspect === 'healthy'}"><span>${count(b.healthy)} healthy people</span><span class="branch-arrow">1% test positive</span><strong>${count(b.falsePositive)}</strong><span>false alarms</span><small>${count(b.healthy - b.falsePositive)} test negative</small></button>
        </div>` : `<div class="population-picture">
          <div class="population-labels"><span><i class="lesson-dot true-dot"></i>${count(s.step === 2 ? b.truePositive : b.sick)} sick${s.step === 2 ? ' + positive' : ''}</span><span><i class="lesson-dot ${s.step === 2 ? 'false-dot' : 'healthy-dot'}"></i>${count(s.step === 2 ? b.falsePositive : b.healthy)} healthy${s.step === 2 ? ' + positive' : ''}</span></div>
          <div class="population-strip" role="img" aria-label="${percent(share)} of this group are sick. Width represents the fraction of people."><span class="true-area" style="width:${share * 100}%"></span><span class="${s.step === 2 ? 'false-area' : 'healthy-area'}" style="width:${100 - share * 100}%"></span></div>
          <div class="population-ruler" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div>
          ${s.step === 2 ? `<div class="posterior-equation"><div class="lesson-fraction"><span>${count(b.truePositive)} sick + positive</span><span>${count(b.truePositive)} + ${count(b.falsePositive)} total positives</span></div><span class="equals">=</span><strong>${percent(b.posterior)}</strong></div>` : `<p class="lesson-caption population-total">Entire rectangle = 10,000 people. Width = share of the population.</p>`}
        </div>`}
        <p class="lesson-observation">${s.step === 0 ? 'Next: test both groups. Healthy people can test positive too.' : s.step === 1 ? (s.inspect === 'healthy' ? `${count(b.healthy)} × 0.01 = ${count(b.falsePositive)} false alarms. A tiny rate can matter when the starting group is large.` : `${count(b.sick)} × 0.99 = ${count(b.truePositive)} real detections. Tap the healthy group to see the other source of positives.`) : s.prior === .01 ? 'The false alarms are as numerous as the real detections. A positive result narrows the odds to 50/50.' : 'Same test. Different starting population. The positive results now contain a much larger share of sick people.'}</p>
      </div>
      <div class="lesson-footer"><button type="button" class="lesson-nav" data-action="previous" ${s.step === 0 ? 'disabled' : ''}>Previous</button><span class="lesson-caption">Hypothetical test · detection 99% · false alarms 1%</span><button type="button" class="lesson-nav lesson-next" data-action="next">${s.step === 2 ? 'Start again' : 'Next step'}</button></div>
    </div>`;
  }

  function entropyMarkup(s) {
    const model = s.mode === 'model';
    const q = model ? (s.q === 'match' ? s.p : Number(s.q)) : s.p;
    const h = entropy(s.p), ce = crossEntropy(s.p, q), kl = Math.max(0, ce - h);
    const outcomes = [{name:'Heads', key:'heads', real:s.p, predicted:q}, {name:'Tails', key:'tails', real:1-s.p, predicted:1-q}];
    const selected = outcomes.find(o => o.key === s.selected);
    return `<div class="lesson lesson-entropy">
      <div class="lesson-tabs" role="group" aria-label="Entropy lesson">
        ${choice('mode', 'entropy', '1. Average surprise', !model)}${choice('mode', 'model', '2. Add a model', model)}
      </div>
      <div class="lesson-options"><span class="lesson-caption">The real coin</span><div class="lesson-choices" role="group" aria-label="Real chance of heads">
        ${[.5,.9,.99].map(p => choice('p', p, p === .5 ? 'Fair coin' : `${percent(p)} heads`, s.p === p)).join('')}
      </div></div>
      ${model ? `<div class="lesson-options model-options"><span class="lesson-caption">The model predicts</span><div class="lesson-choices" role="group" aria-label="Model chance of heads">${[['match','Matches reality'],[.5,'50% heads'],[.1,'10% heads']].map(([q,label]) => choice('q', q, label, s.q === q)).join('')}</div></div>` : ''}
      <div class="surprise-layout">
        <div class="coin-outcomes" role="group" aria-label="Inspect an outcome">
          ${outcomes.map(o => `<button type="button" class="outcome-button ${o.key}" data-action="outcome" data-value="${o.key}" aria-pressed="${s.selected === o.key}"><span class="coin-face" aria-hidden="true">${o.name[0]}</span><strong>${o.name}</strong><span>${percent(o.real)} of real flips</span><span class="outcome-cost">${bits(-Math.log2(o.predicted))} <small>bits of surprise</small></span>${model ? `<span>Model predicts ${percent(o.predicted)}</span>` : ''}</button>`).join('')}
        </div>
        <div class="surprise-average" aria-live="polite" aria-atomic="true">
          <p class="lesson-caption">${model ? 'Cross-entropy' : 'Entropy'} · average per flip</p>
          <div class="surprise-value">${bits(ce)} <span>bits</span></div>
          <p class="weighted-equation"><span>${percent(s.p)} × ${bits(-Math.log2(q))}</span><span>+ ${percent(1-s.p)} × ${bits(-Math.log2(1-q))}</span></p>
          <p>${model ? 'Count outcomes as often as they really happen. Score their surprise using the model\'s guesses.' : 'Each surprise is weighted by how often that outcome happens. Not a plain average of the two costs.'}</p>
        </div>
      </div>
      <p class="lesson-observation" aria-live="polite"><strong>${selected.name}:</strong> ${percent(selected.predicted)} ${model ? 'predicted' : 'real'} probability means −log₂(${selected.predicted.toFixed(2)}) = <strong>${bits(-Math.log2(selected.predicted))} bits</strong>. ${model ? `The model ${selected.predicted >= .5 ? 'expects' : 'does not expect'} this outcome; reality produces it on ${percent(selected.real)} of flips.` : selected.real < .5 ? `Surprising, but it happens only ${percent(selected.real)} of the time.` : selected.real === .5 ? 'With a fair coin, both outcomes carry the same surprise.' : 'Usually expected, so it adds little surprise on each flip.'}</p>
      ${model ? `<div class="loss-decomposition">
        <div class="decomposition-labels"><span><i class="lesson-dot true-dot"></i>Entropy <strong>${bits(h)}</strong></span><span><i class="lesson-dot false-dot"></i>Extra mismatch (KL) <strong>${bits(kl)}</strong></span></div>
        <div class="decomposition-bar" role="img" aria-label="${bits(h)} bits of entropy plus ${bits(kl)} bits of KL equals ${bits(ce)} bits of cross-entropy"><span class="true-area" style="width:${h / ce * 100}%"></span><span class="false-area" style="width:${kl / ce * 100}%"></span></div>
        <p>${kl < 1e-9 ? 'The model matches reality. No extra penalty: cross-entropy equals entropy.' : 'The coin did not change. The extra loss comes from the model expecting the wrong outcomes.'}</p>
      </div>` : `<p class="lesson-takeaway">${s.p === .5 ? 'A fair coin is hard to predict every time: 1 bit per flip.' : `The rare tail is a big surprise, but most flips are predictable. Average surprise falls to ${bits(h)} bits.`}</p>`}
    </div>`;
  }

  function expectationMarkup(s) {
    const weights = dieShapes[s.shape];
    const total = weights.reduce((a,b) => a+b, 0), mean = expectation(weights);
    const chance = weights[s.face - 1] / total;
    return `<div class="lesson lesson-expectation">
      <div class="lesson-options"><span class="lesson-caption">Choose a die</span><div class="lesson-choices" role="group" aria-label="Die distribution">${[['fair','Fair die'],['high','High rolls favored'],['extremes','Only 1 or 6']].map(([shape,label]) => choice('shape',shape,label,s.shape === shape)).join('')}</div></div>
      <div class="expectation-heading"><div><p class="lesson-caption">Expected value</p><div class="surprise-value">${mean.toFixed(1)}</div></div><p>${s.shape === 'extremes' ? 'You can only roll a 1 or a 6. Their average is still 3.5.' : s.shape === 'fair' ? 'All six faces have equal weight. The balance point sits halfway between 3 and 4.' : 'More probability sits on the right, so the balance point moves right too.'}</p></div>
      <div class="die-faces" role="group" aria-label="Tap a face to inspect its contribution">
        ${weights.map((weight,i) => `<button type="button" class="die-face" data-action="face" data-value="${i+1}" aria-label="Face ${i+1}: ${percent(weight / total)} chance. Inspect contribution." aria-pressed="${s.face === i+1}"><span class="die-bar-space"><span class="die-bar" style="height:${weight / Math.max(...weights) * 100}%"></span></span><strong>${i+1}</strong><small>${percent(weight / total)}</small></button>`).join('')}
      </div>
      <div class="balance-rail" aria-hidden="true"><span style="left:${(mean-.5) / 6 * 100}%">▲</span></div>
      <p class="lesson-caption balance-caption">The average is the balance point of these probability weights.</p>
      <p class="lesson-observation" aria-live="polite"><strong>Face ${s.face}:</strong> ${s.face} × (${weights[s.face-1]}/${total}) = ${(s.face*chance).toFixed(3)} toward the average. Tap another face. <strong>Add all six contributions to get ${mean.toFixed(1)}.</strong></p>
      <div class="sampling-experiment"><div class="lesson-choices"><button type="button" class="lesson-choice" data-action="roll" data-value="1">Roll once</button><button type="button" class="lesson-choice" data-action="roll" data-value="100">Roll 100 times</button><button type="button" class="lesson-choice" data-action="reset-rolls">Reset</button></div><p aria-live="polite">${s.rolls ? `<strong>${count(s.rolls)} rolls</strong> · sum ${count(s.sum)} ÷ ${count(s.rolls)} = <strong>${(s.sum/s.rolls).toFixed(2)} observed average</strong>.${s.rolls === 1 ? ` You rolled a ${s.sum}.` : ''}` : 'Now try repeated rolls. The observed average will fluctuate around the expected value.'}</p></div>
    </div>`;
  }

  function update(instance, action, value) {
    const s = instance.state;
    if (action === 'step') s.step = Number(value);
    if (action === 'next') s.step = (s.step + 1) % 3;
    if (action === 'previous') s.step = Math.max(0, s.step - 1);
    if (action === 'prior') s.prior = Number(value);
    if (action === 'inspect') s.inspect = value;
    if (action === 'p') s.p = Number(value);
    if (action === 'q') s.q = value === 'match' ? value : Number(value);
    if (action === 'outcome') s.selected = value;
    if (action === 'mode') s.mode = value;
    if (action === 'face') s.face = Number(value);
    if (action === 'shape') { s.shape = value; s.rolls = 0; s.sum = 0; }
    if (action === 'reset-rolls') { s.rolls = 0; s.sum = 0; }
    if (action === 'roll') {
      const weights = dieShapes[s.shape], total = weights.reduce((a,b) => a+b, 0);
      for (let i=0; i<Number(value); i++) {
        let sample = Math.random() * total;
        let face = 0;
        while (face < 5 && sample >= weights[face]) { sample -= weights[face]; face++; }
        s.sum += face+1;
        s.rolls++;
      }
    }
    paint(instance);
  }
  function paint(instance) {
    const markup = {bayes:bayesMarkup, entropy:entropyMarkup, expectation:expectationMarkup}[instance.kind](instance.state);
    instance.roots.forEach(root => {
      const active = root.contains(document.activeElement) ? document.activeElement : null;
      const action = active?.dataset.action, value = active?.dataset.value;
      root.innerHTML = markup;
      if (action) {
        const next = [...root.querySelectorAll('[data-action]')].find(el => el.dataset.action === action && el.dataset.value === value);
        next?.focus({preventScroll:true});
      }
    });
  }
  function render(kind, root) {
    if (!instances.has(kind)) instances.set(kind, {kind, state:kind === 'bayes' ? {prior:.01, step:0, inspect:'sick'} : kind === 'expectation' ? {shape:'fair',face:6,rolls:0,sum:0} : {p:.5, q:'match', mode:'entropy', selected:'heads'}, roots:new Set()});
    const instance = instances.get(kind);
    if (!instance.roots.has(root)) {
      instance.roots.add(root);
      root.addEventListener('click', event => {
        const button = event.target.closest('[data-action]');
        if (!button || button.disabled) return;
        update(instance, button.dataset.action, button.dataset.value);
      });
    }
    paint(instance);
  }
  const api = {has:kind => ['bayes','entropy','expectation'].includes(kind), render, bayes, entropy, crossEntropy, expectation};
  if (typeof window !== 'undefined') window.AtelierLessons = api;
  if (typeof module !== 'undefined') module.exports = api;
})();
