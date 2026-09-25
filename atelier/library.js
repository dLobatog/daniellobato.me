const guesses = document.querySelectorAll('[data-guess]');
guesses.forEach(button => button.addEventListener('click', () => {
  guesses.forEach(guess => guess.setAttribute('aria-pressed', String(guess === button)));
  document.querySelector('.opening-answer').hidden = false;
  document.querySelector('[data-guess-feedback]').innerHTML = `${button.dataset.guess === '50' ? 'Exactly.' : 'Surprisingly, it is 50%.'} In 10,000 people, <strong>99 of the 198 positive results</strong> come from sick people. The other half are false alarms.`;
}));

const search = document.querySelector('#chapter-search');
const cards = [...document.querySelectorAll('.route-card')];
const synonyms = {
  'foundations.html': 'bayes entropy cross entropy probability distribution expectation loss',
  'neural-network-basics.html': 'neuron activation sigmoid softmax backprop chain rule gradient descent optimizer learning rate',
  'gbdt-tabular.html': 'tree trees boosting gbdt histogram split',
  'adaptation-serving.html': 'lora peft quantization fine tuning knowledge distillation',
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
