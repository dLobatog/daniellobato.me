const guesses = document.querySelectorAll('[data-guess]');
guesses.forEach(button => button.addEventListener('click', () => {
  guesses.forEach(guess => guess.setAttribute('aria-pressed', String(guess === button)));
  document.querySelector('.opening-answer').hidden = false;
  document.querySelector('[data-guess-feedback]').innerHTML = `${button.dataset.guess === '31' ? 'Exactly: about 31%.' : 'Surprisingly, only about 31%.'} Among 10,000 emails, <strong>90 of the 288 flagged messages</strong> are spam. The much larger legitimate population creates more false alarms than real detections.`;
}));

const search = document.querySelector('#chapter-search');
const cards = [...document.querySelectorAll('.route-card')];
const synonyms = {
  'foundations.html': 'bayes entropy cross entropy probability distribution expectation loss',
  'neural-network-basics.html': 'neuron activation sigmoid softmax backprop chain rule gradient descent optimizer learning rate',
  'gbdt-tabular.html': 'tree trees boosting gbdt histogram split',
  'adaptation-serving.html': 'lora peft quantization fine tuning knowledge distillation',
  'linear-algebra.html': 'vector dot product matrix multiplication eigenvalue eigenvector svd singular value rank projection',
  'deep-learning.html': 'initialization gradients vanishing exploding normalization batchnorm layernorm residual flash attention',
  'classical-ml-stats.html': 'mle map maximum likelihood bayesian bias variance l1 l2 regularization',
  'transformers-rag.html': 'bpe tokenization embeddings positional rope transformer architecture attention query key value kv cache retrieval augmented generation',
  'recommendation-depth.html': 'matrix factorization two tower learning to rank lambdarank pairwise listwise recommendation personalization',
  'systems-retrieval.html': 'candidate retrieval ranking cold start exploration exposure recommendations personalization',
  'metrics-eval.html': 'precision recall threshold calibration temperature scaling brier ndcg ranking metrics',
  'data-features.html': 'point in time leakage availability timestamps missing values feature shift data splits',
  'production-systems.html': 'training serving skew offline online evaluation experiments ab test cohort reweighting data concept drift delayed labels',
  'reinforcement-learning.html': 'markov mdp bellman value policy td temporal difference q learning sarsa dqn bandit exploration',
  'generative-and-rl.html': 'diffusion denoising ddim bandit exploration exploitation',
  'alignment-depth.html': 'dpo grpo group relative policy optimization reward kl clipping alignment preference guidance cfg reward hacking',
};
search.addEventListener('input', () => {
  const terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
  let visible = 0;
  cards.forEach(card => {
    const file = card.querySelector('a').getAttribute('href').split('/').pop();
    const text = `${card.textContent} ${synonyms[file] || ''}`.toLowerCase();
    card.hidden = !terms.every(term => text.includes(term));
    if (!card.hidden) visible++;
  });
  const status = document.querySelector('.search-status');
  status.hidden = terms.length === 0;
  status.textContent = visible ? `${visible} matching ${visible === 1 ? 'chapter' : 'chapters'}` : 'No matching chapters. Try a broader term, such as "attention" or "loss".';
});
