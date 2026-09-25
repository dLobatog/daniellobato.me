/* An actual 9 -> 2 -> 1 image classifier, shared by four views of its computation. */
(() => {
  'use strict';
  const KINDS = ['neuron','forward-pass','chain-rule','backprop'];
  const images = {
    vertical: [0,1,0,0,1,0,0,1,0],
    horizontal: [0,0,0,1,1,1,0,0,0],
    cross: [0,1,0,1,1,1,0,1,0],
  };
  const initialParameters = () => ({
    W: [[-.2,.8,-.2,-.2,.8,-.2,-.2,.8,-.2],[-.2,-.2,-.2,.8,.8,.8,-.2,-.2,-.2]],
    b: [-.1,-.1], v: [.5,-.5], c: 0,
  });
  const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  const dot = (a,b) => a.reduce((sum,x,i) => sum + x*b[i],0);
  const softplus = z => Math.max(z,0) + Math.log1p(Math.exp(-Math.abs(z)));
  function forward(x, p, target = 1) {
    if (x.length !== 9 || x.some(v=>!Number.isFinite(v)) || ![0,1].includes(target)) throw new RangeError('Expected nine finite pixels and a binary label');
    const z = p.W.map((w,j)=>dot(w,x)+p.b[j]);
    const a = z.map(v=>Math.max(0,v));
    const score = dot(p.v,a)+p.c;
    const probability = sigmoid(score);
    return {z,a,score,probability,loss:softplus(score)-target*score,target};
  }
  function backward(x,p,target=1) {
    const f = forward(x,p,target);
    const delta = f.probability-target;
    const hidden = p.v.map((v,j)=>delta*v*(f.z[j]>0?1:0));
    return {delta,hidden,W:hidden.map(d=>x.map(pixel=>d*pixel)),b:[...hidden],v:f.a.map(a=>delta*a),c:delta};
  }
  function trainStep(x,p,target=1,rate=.2) {
    if (!Number.isFinite(rate) || rate<0) throw new RangeError('Learning rate must be finite and nonnegative');
    const gradient = backward(x,p,target);
    const next = {W:p.W.map((w,j)=>w.map((v,i)=>v-rate*gradient.W[j][i])),b:p.b.map((b,j)=>b-rate*gradient.b[j]),v:p.v.map((v,j)=>v-rate*gradient.v[j]),c:p.c-rate*gradient.c};
    return {before:forward(x,p,target),after:forward(x,next,target),gradient,parameters:next};
  }
  const f = (n,d=3) => Number(n.toFixed(d)).toString();
  const pct = n => `${f(n*100,1)}%`;
  const signed = n => `${n>0?'+':''}${f(n)}`;
  const defaultPass = forward(images.vertical,initialParameters());

  const math = {
    neuron: {
      title:'One hidden neuron reads nine pixels',
      formula:['z_j=\\sum_{i=1}^{9}w_{ji}x_i+b_j','a_j=\\max(0,z_j)','z_1=3(0.8)-0.1=2.3,\\quad a_1=2.3'],
      note:'These are illustrative starting weights. Positive weights favor lit pixels; negative weights oppose them. ReLU keeps positive scores and maps negative scores to zero.',
      annotations:[['x_i','Brightness of pixel i: 0 is dark, 1 is lit.','The three middle-column pixels are lit in the vertical-stroke input.'],['w_{ji}','Connection from pixel i to hidden neuron j.','Neuron 1 assigns +0.8 to the middle column and -0.2 elsewhere.'],['b_j','A learned offset before activation.','The -0.1 bias means an empty image gives z=-0.1 and activation 0.'],['a_j','The neuron output after ReLU.','Three matching lit pixels give an activation of 2.3, not a probability.']],
    },
    'forward-pass': {
      title:'Pixels to hidden activations to a class probability',
      formula:['\\mathbf a=\\operatorname{ReLU}(W\\mathbf x+\\mathbf b)','s=\\mathbf v^\\top\\mathbf a+c','p=\\sigma(s)=\\frac{1}{1+e^{-s}}','L=-y\\log p-(1-y)\\log(1-p)'],
      note:'For the vertical input, a=(2.3,0.3), s=0.5(2.3)-0.5(0.3)=1, p=0.731, and L=-log(0.731)=0.313 nats.',
      annotations:[['W,\\mathbf b','The hidden layer: a 2-by-9 matrix and two biases.','Each row of W scores a different arrangement of the same nine pixels.'],['\\mathbf a','Two intermediate activations.','The vertical input activates neuron 1 more strongly: 2.3 versus 0.3.'],['\\mathbf v,c','Output-layer weights and bias.','Initially v=(0.5,-0.5), so neuron 1 supports the vertical class and neuron 2 opposes it.'],['p,y,L','Predicted probability, training label, and binary cross-entropy.','For a vertical label y=1, assigning probability 0.731 incurs loss 0.313.']],
    },
    'chain-rule': {
      title:'Follow a pixel weight all the way to the loss',
      formula:['\\frac{\\partial L}{\\partial w_{ji}}=(p-y)\\,v_j\\,\\operatorname{ReLU}\'(z_j)\\,x_i','\\frac{\\partial L}{\\partial w_{1,2}}=(0.731-1)(0.5)(1)(1)\\approx-0.1345'],
      note:'This simplified sigmoid-plus-cross-entropy derivative already includes the output sigmoid derivative. Do not multiply by p(1-p) again. At z=0 we use the usual ReLU backward convention of zero.',
      annotations:[['p-y','Loss derivative with respect to the output logit.','The vertical image is underconfident: 0.731-1=-0.269.'],['v_j','How the hidden activation changes the output logit.','Neuron 1 has output weight +0.5; neuron 2 has -0.5.'],["\\operatorname{ReLU}'(z_j)",'How much gradient passes the hidden activation.','Both current hidden scores are positive, so this factor is 1.'],['x_i','How much this pixel weight changes the hidden score.','A lit pixel contributes a factor of 1; a dark pixel contributes 0, making this connection gradient zero.']],
    },
    backprop: {
      title:'Reuse derivatives backward, then update every parameter',
      formula:['\\delta_{out}=p-y','\\boldsymbol\\delta_h=(\\delta_{out}\\mathbf v)\\odot\\mathbf 1[\\mathbf z>0]','\\nabla_W L=\\boldsymbol\\delta_h\\mathbf x^\\top','\\theta_{new}=\\theta-\\eta\\nabla_\\theta L'],
      note:'All gradients are computed using the old parameters. Here eta=0.2. Improvement on one training image is not evidence of generalization.',
      annotations:[['\\delta_{out}','Sensitivity of loss to the output logit.','For the vertical input, it is -0.269.'],['\\boldsymbol\\delta_h','Sensitivities arriving at the two hidden scores.','Multiplying by output weights gives approximately (-0.1345,+0.1345).'],['\\nabla_W L','One gradient for each pixel-to-hidden connection.','For neuron 1, lit-pixel connections have gradient -0.1345; dark-pixel connections have gradient 0.'],['\\eta,\\theta','Learning rate and all 23 trainable parameters.','With eta=0.2, a matching pixel weight changes from 0.8 to about 0.8269.']],
    },
  };
  const code = `import torch\nimport torch.nn.functional as F\n\nx = torch.tensor([0.,1.,0., 0.,1.,0., 0.,1.,0.])\nW = torch.tensor([[-.2,.8,-.2, -.2,.8,-.2, -.2,.8,-.2],\n                  [-.2,-.2,-.2, .8,.8,.8, -.2,-.2,-.2]],\n                 requires_grad=True)\nb = torch.tensor([-.1, -.1], requires_grad=True)\nv = torch.tensor([.5, -.5], requires_grad=True)\nc = torch.tensor(0., requires_grad=True)\n\na = torch.relu(W @ x + b)\nlogit = v @ a + c\nloss = F.binary_cross_entropy_with_logits(logit, torch.tensor(1.))\nloss.backward()  # compute gradients; parameters are unchanged\n\nwith torch.no_grad():\n    for parameter in (W, b, v, c):\n        parameter -= 0.2 * parameter.grad\n        parameter.grad.zero_()`;
  const content = {
    neuron: {title:'A neuron responds to a pattern in its inputs',summary:'Nine pixels enter. A weighted sum scores their arrangement. An activation decides what the next layer receives.',what:'The weight map is the neuron\'s <strong>pattern of sensitivity</strong>. A lit pixel on a positive weight raises its score; a lit pixel on a negative weight lowers it.',why:'Look at the input beside the weights. You can see why a vertical stroke activates one neuron more than a horizontal stroke does.',interview:'An affine map followed by a nonlinear activation is the basic building block. Hidden activations are features, not necessarily probabilities or human-interpretable concepts.',details:['This is a fully connected 9 → 2 → 1 classifier, not a convolution: every hidden neuron has a separate weight for every pixel.','The example weights deliberately make the mechanism visible. Real training may learn distributed features with no neat name.','Changing the image changes activations; only a training update changes the weights.'],quiz:{prompt:'A dark pixel has value 0. What happens if you increase only its connection weight?',options:[{text:'The current prediction stays unchanged.',correct:true,explanation:'That contribution is weight × 0. The weight can matter on a different image where this pixel is lit.'},{text:'The neuron becomes more active on this image.',correct:false,explanation:'The weight is multiplied by the input. A zero input contributes nothing.'},{text:'The pixel itself becomes brighter.',correct:false,explanation:'Weights are parameters, not input pixels. Changing a weight does not edit the image.'}]}},
    'forward-pass': {title:'Watch an image become a prediction',summary:'Trace pixels through a hidden layer, a class score, and a probability. The parameters stay fixed throughout.',what:'The <strong>forward pass</strong> evaluates the current network. Each hidden neuron sees the same image with different weights; the output layer combines their activations.',why:'A network is a composition of ordinary operations. Its prediction depends on the intermediate features, not a direct lookup of the label.',interview:'Forward computes and caches activations. During training, a loss compares the output with the label; inference does not require a known label.',details:['The output sigmoid models P(vertical | image). The horizontal probability is its complement.','A cross activates both hidden neurons equally. The opposing output weights cancel, giving 50/50: uncertainty is visible in the network, not just its final number.','This tiny synthetic example does not claim recognition of arbitrary handwriting or translation invariance.'],quiz:{prompt:'Why does the cross-shaped input initially get a 50/50 prediction?',options:[{text:'Both hidden activations are equal, and their output weights oppose each other.',correct:true,explanation:'The score is 0.5a₁ - 0.5a₂. Equal activations cancel, and sigmoid(0)=0.5.'},{text:'Every network must output 50/50 on unfamiliar data.',correct:false,explanation:'Neural networks can be confidently wrong on unfamiliar data. Here the equality follows from these particular weights.'},{text:'A forward pass updates the weights until the classes tie.',correct:false,explanation:'Forward evaluation does not update any weights.'}]}},
    'chain-rule': {title:'Trace a gradient through the actual network',summary:'Select a pixel connection. Follow its influence through the hidden activation, output score, and loss.',what:'The <strong>chain rule</strong> multiplies local derivatives along a computation path. A zero anywhere on this path blocks the gradient for that connection on this example.',why:'You can explain a gradient\'s sign and magnitude by inspecting the operations that produced it, then check the result with a tiny numerical perturbation.',interview:'For this binary classifier with cross-entropy, the logit derivative is p-y. Hidden-layer derivatives multiply by the readout weight, activation derivative, and input. Multiple paths contribute a sum.',details:['Select a dark pixel to see an input factor of zero. This says nothing about that connection on other images.','A negative hidden score would also block this path through ReLU. The ordinary derivative at exactly zero is undefined; this implementation uses zero.','The numerical probe changes just the selected weight. It verifies a local approximation, not a guarantee for arbitrarily large steps.'],quiz:{prompt:'The input pixel is lit, but its hidden neuron has a negative pre-activation. What is this path\'s gradient through ReLU?',options:[{text:'Zero, because ReLU is locally flat there.',correct:true,explanation:'The ReLU derivative is zero for a negative pre-activation, so the product of local derivatives is zero.'},{text:'Always p-y.',correct:false,explanation:'p-y is the output-logit derivative. Earlier connections require the remaining chain factors too.'},{text:'The learning rate.',correct:false,explanation:'The learning rate scales an optimizer update; it is not a derivative.'}]}},
    backprop: {title:'An error changes the network, not the input',summary:'Send the loss derivative backward, update the connections, and run the same image forward again.',what:'<strong>Backpropagation computes gradients</strong> for all parameters by reusing intermediate derivatives. The <strong>optimizer</strong> then applies a separate update.',why:'Watch which connections move and which do not. A training example changes the model\'s response to a pattern, not the pixels or their label.',interview:'For a dense layer, weight gradients are outer products of the upstream derivative and the input activation. Compute all gradients at the same parameter state before updating.',details:['Here one SGD update changes both hidden-layer weights and the output layer, with learning rate 0.2.','Training on a single image repeatedly can overfit. Test the other images after an update; lower training loss alone is not a sufficient evaluation.','An activation changing during inference is not learning. Learning requires a change to trainable parameters.'],quiz:{prompt:'After backward(), but before the optimizer update, what happens to the probability for the same image?',options:[{text:'Nothing. Gradients were computed, but the weights are unchanged.',correct:true,explanation:'Run the same forward calculation with the same parameters and inputs: it produces the same probability.'},{text:'It becomes 100% correct.',correct:false,explanation:'Backward does not overwrite predictions or labels. Even a useful update need not remove all error.'},{text:'The input pixels change to reduce the loss.',correct:false,explanation:'Ordinary supervised training updates model parameters, not the training image.'}]}}
  };
  for (const kind of KINDS) Object.assign(content[kind],{math:math[kind],code:{title:'The same classifier in PyTorch',lang:'python',snippet:code},controls:[],presets:[],geometry:null});
  const steps = {
    neuron:['Multiply each pixel by its weight','Add the contributions and bias','Apply the activation'],
    'forward-pass':['Read nine pixels','Compute two hidden features','Combine features into a probability','Compare with the training label'],
    'chain-rule':['Select a connection','Trace the local derivatives','Multiply the path','Check with a small perturbation'],
    backprop:['Forward: measure the loss','Backward: compute gradients','Optimizer: apply the update','Forward again: compare'],
  };
  function initial() { return {pixels:[...images.vertical],target:1,preset:'vertical',selected:0,pixel:1,step:0,parameters:initialParameters(),round:0}; }
  function reduce(kind,s,action,value) {
    if (action==='reset') return initial();
    if (action==='step') return {...s,step:Math.max(0,Math.min(steps[kind].length-1,Number(value)))};
    if (action==='next') return {...s,step:Math.min(steps[kind].length-1,s.step+1)};
    if (action==='previous') return {...s,step:Math.max(0,s.step-1)};
    if (action==='image' && images[value]) return {...s,pixels:[...images[value]],preset:value,target:value==='horizontal'?0:1};
    if (action==='pixel') { const i=Number(value);if(!Number.isInteger(i)||i<0||i>8)return s;return {...s,pixels:s.pixels.map((p,j)=>j===i?1-p:p),pixel:i,preset:'edited'}; }
    if (action==='weight') {const i=Number(value);return i>=0&&i<9?{...s,pixel:i}:s;}
    if (action==='neuron' && [0,1].includes(Number(value))) return s.selected===Number(value)?s:{...s,selected:Number(value)};
    if (action==='target') return {...s,target:Number(value)===1?1:0};
    if (action==='train' && kind==='backprop') return {...s,parameters:trainStep(s.pixels,s.parameters,s.target).parameters,round:s.round+1,step:0};
    return s;
  }
  const btn = (action,value,text,selected=false,extra='') => `<button type="button" data-action="${action}" data-value="${value}" ${selected?'aria-pressed="true"':''} ${extra}>${text}</button>`;
  function pixels(s) {
    return `<div class="nn-pixels" role="group" aria-label="Input image. Select a pixel to toggle it">${s.pixels.map((p,i)=>btn('pixel',i,p?'1':'0',false,`class="nn-pixel ${p?'is-lit':''}" aria-label="Pixel ${i+1}: ${p?'lit':'dark'}. Toggle"`)).join('')}</div>`;
  }
  function weights(s,gradient=false) {
    const g=backward(s.pixels,s.parameters,s.target);
    const values=gradient?g.W[s.selected]:s.parameters.W[s.selected];
    return `<div class="nn-weights" role="group" aria-label="${gradient?'Gradients':'Weights'} of neuron ${s.selected+1}">${values.map((w,i)=>btn('weight',i,f(w),i===s.pixel,`class="nn-weight ${w>0?'positive':w<0?'negative':''}" aria-label="Pixel ${i+1}, ${gradient?'gradient':'weight'} ${f(w)}. Inspect connection"`)).join('')}</div>`;
  }
  function network(s,pass,showGradient=false,reveal=2) {
    const g=backward(s.pixels,s.parameters,s.target);
    return `<div class="nn-network">
      <div class="nn-input-column"><span class="nn-layer-label">Input · 9 pixels</span>${pixels(s)}<small>Click a pixel to change the image.</small></div>
      <div class="nn-wire" aria-hidden="true">→</div>
      <div class="nn-hidden-column"><span class="nn-layer-label">Hidden · 2 ReLU neurons</span>${[0,1].map(j=>btn('neuron',j,`<span>Neuron ${j+1}</span><strong>${reveal>=1?f(pass.a[j]):'?'}</strong><small>${reveal<1?'Not computed yet':showGradient?'∂L/∂z = '+f(g.hidden[j]):'z = '+f(pass.z[j])}</small>`,s.selected===j,'class="nn-hidden"')).join('')}<small>Inspect a neuron to see its weights.</small></div>
      <div class="nn-wire" aria-hidden="true">→</div>
      <div class="nn-output-column"><span class="nn-layer-label">Output · sigmoid</span><div class="nn-output"><span>Vertical stroke</span><strong>${reveal>=2?pct(pass.probability):'?'}</strong><span>${reveal>=2?'Horizontal: '+pct(1-pass.probability):'Waiting for the hidden features'}</span><small>${reveal>=2?'logit = '+f(pass.score):'The next layer uses their activations.'}</small></div></div>
    </div>`;
  }
  const narratives = {
    neuron:['A neuron sees numbers, not a named object. Match each pixel with the weight in the same position.','Lit pixels on positive weights add evidence. Lit pixels on negative weights subtract it. Dark pixels contribute zero.','ReLU passes positive scores unchanged and maps negative scores to zero. The next layer receives this activation, not a class label.'],
    'forward-pass':['The image is nine input values. The same pixels feed both hidden neurons.','Different weight maps make the same image produce different activations. Inspect either neuron.','The output weights favor neuron 1 and oppose neuron 2. A sigmoid turns their combined score into a binary probability.','The label is used to compute loss, not to produce the prediction. No weight has changed.'],
    'chain-rule':['Choose a hidden neuron, then one of its nine pixel connections. The highlighted weight is the parameter we will differentiate.','Trace how this weight changes the hidden score, activation, output logit, and finally the loss.','Multiply the local rates. Select a dark pixel: its zero input blocks the gradient for this connection.','Nudge just this weight by 0.001 and run a fresh forward pass. Compare the measured loss change with the derivative prediction.'],
    backprop:['The input and label are fixed. The current network makes a prediction and incurs a loss.','The output derivative flows through both hidden neurons. Each pixel weight receives the hidden derivative multiplied by that pixel.','SGD subtracts 0.2 times each gradient. Lit-pixel connections can move; zero-input connections have zero gradient on this image.','Run the same image through the updated network. Its prediction changes because the parameters changed, not because the image changed.'],
  };
  function render(kind,s) {
    const p=s.parameters, result=forward(s.pixels,p,s.target), update=trainStep(s.pixels,p,s.target), g=update.gradient;
    const shown=kind==='backprop'&&s.step===3?update.after:result;
    const shownState=kind==='backprop'&&s.step===3?{...s,parameters:update.parameters}:s;
    const j=s.selected,i=s.pixel;
    let detail='';
    if (kind==='neuron' || kind==='forward-pass') {
      detail=`<div class="nn-detail"><div><h4>Neuron ${j+1}: connection weights</h4>${weights(s)}<p class="ml-note">Blue adds to the score. Orange subtracts.</p></div><div class="nn-calculation"><h4>Selected connection</h4><p class="nn-equation">${s.pixels[i]} × ${f(p.W[j][i])} = <strong>${f(s.pixels[i]*p.W[j][i])}</strong></p><p>Pixel ${i+1} contributes ${f(s.pixels[i]*p.W[j][i])} to this neuron.</p><p class="nn-equation">Σ contributions ${f(dot(p.W[j],s.pixels))}<br>+ bias ${f(p.b[j])}<br>= score <strong>${f(result.z[j])}</strong></p><p>ReLU(${f(result.z[j])}) = <strong>${f(result.a[j])}</strong></p></div></div>`;
      if (kind==='forward-pass'&&s.step>=2) detail+=`<p class="ml-inspector"><strong>Output score:</strong> ${f(p.v[0])} × ${f(result.a[0])} + (${f(p.v[1])}) × ${f(result.a[1])} + ${f(p.c)} = ${f(result.score)}. Sigmoid gives ${pct(result.probability)} for vertical.</p>`;
      if(kind==='forward-pass'&&s.step===3)detail+=`<p class="ml-inspector"><strong>Label: ${s.target?'vertical':'horizontal'}.</strong> The probability assigned to the correct class is ${pct(s.target?result.probability:1-result.probability)}. Its negative natural log is <strong>${f(result.loss)} nats of loss</strong>.</p>`;
    } else if(kind==='chain-rule') {
      const factors=[{name:'Loss → logit',value:g.delta,why:'p − label'},{name:'Logit → activation',value:p.v[j],why:`Output weight v${j+1}`},{name:'Activation → score',value:result.z[j]>0?1:0,why:'ReLU local slope'},{name:'Score → weight',value:s.pixels[i],why:`Input pixel ${i+1}`}];
      const eps=.001,nudged={...p,W:p.W.map((w,k)=>w.map((v,l)=>v+(k===j&&l===i?eps:0)))};
      const actual=forward(s.pixels,nudged,s.target).loss-result.loss;
      detail=`<div class="nn-detail"><div><h4>Select a pixel connection</h4>${weights(s)}</div><div>${s.step===0?`<h4>One weight, one path to the loss</h4><p class="nn-equation">weight → hidden score → activation → output score → loss</p><p>Selected weight: <strong>${f(p.W[j][i])}</strong>. Pixel ${i+1} is ${s.pixels[i]?'lit, so this connection contributes to the hidden score.':'dark, so this connection currently contributes zero.'}</p><p class="ml-note">Next, inspect how sensitive each link is.</p>`:`<h4>Local derivatives along this path</h4><div class="nn-factors">${factors.map(t=>`<div><span>${t.name}</span><strong>${f(t.value)}</strong><small>${t.why}</small></div>`).join('')}</div>${s.step>=2?`<p class="nn-equation">∂L/∂w = ${factors.map(t=>f(t.value)).join(' × ')} = <strong>${f(g.W[j][i],4)}</strong></p>`:''}`}</div></div>`;
      if(s.step===3)detail+=`<div class="nn-probe"><div><span>Gradient predicts Δloss</span><strong>${f(g.W[j][i]*eps,6)}</strong></div><div><span>Recalculation gives Δloss</span><strong>${f(actual,6)}</strong></div><p class="ml-note">Only this connection changed, by +0.001. The approximation becomes more accurate for smaller perturbations.</p></div>`;
    } else {
      const next=update.parameters;
      detail=s.step===0?`<div class="ml-inspector"><strong>Correct class: ${s.target?'vertical':'horizontal'}.</strong><p>The network gives it ${pct(s.target?result.probability:1-result.probability)} probability. Loss = −ln(${f(s.target?result.probability:1-result.probability,4)}) = <strong>${f(result.loss)} nats</strong>.</p><p>Next, trace how changing the weights could lower this loss.</p></div>`:`<div class="nn-detail"><div><h4>Weight gradients · neuron ${j+1}</h4>${weights(s,true)}<p class="ml-note">The shape matches the input: a dark pixel gives a zero weight gradient.</p></div><div><h4>Selected pixel connection ${i+1}</h4><p class="nn-equation">gradient = ${f(g.hidden[j])} × ${s.pixels[i]}<br>= <strong>${f(g.W[j][i],4)}</strong></p><p>Its value before the update is <strong>${f(p.W[j][i],4)}</strong>.</p>${s.step>=2?`<p class="nn-equation">${f(p.W[j][i],4)} − 0.2 × (${f(g.W[j][i],4)})<br>= <strong>${f(next.W[j][i],4)}</strong></p>`:''}</div></div>`;
      if(s.step===3)detail+=`<div class="nn-probe"><div><span>Before · correct-class probability</span><strong>${pct(s.target?result.probability:1-result.probability)}</strong><small>Loss ${f(result.loss)}</small></div><div><span>After one SGD update</span><strong>${pct(s.target?update.after.probability:1-update.after.probability)}</strong><small>Loss ${f(update.after.loss)}</small></div><p class="ml-note">This is one training image, not a test-set result. ${s.round+1} update${s.round?'s':''} shown.</p></div><div class="ml-controls">${btn('train','','Keep these weights; take another step')}</div>`;
    }
    return `<header class="ml-heading"><p class="ml-kicker">A small image classifier · 9 inputs → 2 hidden units → 1 probability</p><h3 class="ml-question">${steps[kind][s.step]}</h3><p class="ml-note">${narratives[kind][s.step]}</p></header>
      <nav class="ml-step-nav" aria-label="Computation steps">${btn('previous','','Previous',false,s.step===0?'disabled':'')}<span>${s.step+1} / ${steps[kind].length}</span>${btn('next','','Next',false,s.step===steps[kind].length-1?'disabled':'')}</nav>
      <div class="ml-controls" aria-label="Input images">${btn('image','vertical','Vertical stroke',s.preset==='vertical')}${btn('image','horizontal','Horizontal stroke',s.preset==='horizontal')}${btn('image','cross','Ambiguous cross',s.preset==='cross')}${btn('reset','','Reset network')}</div>
      ${network(shownState,shown,kind==='backprop'&&s.step===1,kind==='forward-pass'?s.step:2)}
      ${kind==='backprop'?`<div class="ml-controls"><span>Training label:</span>${btn('target',1,'Vertical',s.target===1)}${btn('target',0,'Horizontal',s.target===0)}</div>`:''}
      ${kind==='forward-pass'&&s.step===0?'':detail}
      <details class="nn-caveat"><summary>About this small network</summary><p>Starting weights are chosen to expose the mechanism, not trained image-recognition results. This is a fully connected network, not a convolution. Hidden units in larger networks need not represent a single named feature.</p></details>`;
  }
  const api={images,initialParameters,sigmoid,forward,backward,trainStep,initial,reduce,content,defaultPass};
  if(typeof module!=='undefined')module.exports=api;
  if(typeof window!=='undefined')window.AtelierLab.createModule({id:'neural-mechanisms',content,initial,render,reduce});
})();
