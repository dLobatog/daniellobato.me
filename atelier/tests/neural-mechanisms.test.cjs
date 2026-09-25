const test=require('node:test');
const assert=require('node:assert/strict');
const n=require('../neural-mechanisms.js');
const near=(a,b,e=1e-7)=>assert.ok(Math.abs(a-b)<e,`${a} != ${b}`);
test('Actual classifier exposes the stated hidden activations and probability',()=>{
  const f=n.forward(n.images.vertical,n.initialParameters());
  near(f.z[0],2.3);near(f.z[1],.3);near(f.score,1);near(f.probability,n.sigmoid(1));
  near(n.forward(n.images.horizontal,n.initialParameters(),0).probability,1-f.probability);
  near(n.forward(n.images.cross,n.initialParameters()).probability,.5);
});
test('Analytic gradients match finite differences for all 23 parameters',()=>{
  for(const image of Object.values(n.images)){
    const p=n.initialParameters(),g=n.backward(image,p),eps=1e-5;
    const keys=[...p.W.flatMap((w,j)=>w.map((_,i)=>['W',j,i])),...p.b.map((_,i)=>['b',i]),...p.v.map((_,i)=>['v',i]),['c']];
    for(const path of keys){
      const shifted=amount=>{const q=structuredClone(p);let obj=q;for(const k of path.slice(0,-1))obj=obj[k];obj[path.at(-1)]+=amount;return n.forward(image,q).loss;};
      let analytic=g;for(const k of path)analytic=analytic[k];
      near((shifted(eps)-shifted(-eps))/(2*eps),analytic);
    }
  }
});
test('Dark pixels have zero weight gradients, not zero bias gradients',()=>{
  const g=n.backward(n.images.vertical,n.initialParameters());
  near(g.W[0][0],0);assert.ok(Math.abs(g.b[0])>.1);assert.ok(g.W[0][1]<0);
});
test('One simultaneous SGD update reduces loss without mutating old weights',()=>{
  for(const [name,y] of [['vertical',1],['horizontal',0]]){
    const p=n.initialParameters(),before=structuredClone(p),u=n.trainStep(n.images[name],p,y);
    assert.ok(u.after.loss<u.before.loss);assert.deepEqual(p,before);
    near(u.parameters.W[0][0],p.W[0][0]-.2*u.gradient.W[0][0]);
  }
});
test('Finite probe matches the local derivative; hidden ReLU gate blocks gradients',()=>{
  const p=n.initialParameters(),x=n.images.vertical,g=n.backward(x,p),eps=.001,q=structuredClone(p);
  q.W[0][1]+=eps;assert.ok(Math.abs(n.forward(x,q).loss-n.forward(x,p).loss-g.W[0][1]*eps)<1e-6);
  p.b[0]=-100;assert.ok(n.backward(x,p).W[0].every(v=>v===0));
});
test('Navigation and input edits are bounded, training is a separate state action',()=>{
  let s=n.initial();s=n.reduce('neuron',s,'pixel','0');assert.equal(s.pixels[0],1);assert.deepEqual(s.parameters,n.initialParameters());
  assert.equal(n.reduce('neuron',s,'pixel','99'),s);
  assert.equal(n.reduce('backprop',s,'previous').step,0);
  const u=n.reduce('backprop',s,'train');assert.equal(u.round,1);assert.notDeepEqual(u.parameters,s.parameters);
});
test('Each concept has matching annotations, runnable code and one correct check',()=>{
  for(const c of Object.values(n.content)){
    assert.ok(c.math.annotations.every(a=>a.length===3&&a.every(Boolean)));
    assert.match(c.code.snippet,/loss.backward/);assert.equal(c.quiz.options.filter(q=>q.correct).length,1);
    assert.doesNotMatch(JSON.stringify(c),/kilometer|delivery|min\/km/);
  }
});
test('Pixel edit exposes its exact contribution and preserves a before image',()=>{
  const s=n.initial(),next=n.reduce('neuron',s,'pixel','1');
  assert.deepEqual(next.beforePixels,s.pixels);
  near(n.forward(s.pixels,s.parameters).z[0]-n.forward(next.pixels,next.parameters).z[0],.8);
  const html=n.render('neuron',next);
  assert.match(html,/Score: 2.3 → 1.5/);
  assert.ok(html.indexOf('nn-contributions')<html.indexOf('<details'));
});
test('Chain rule shows a zero factor in the stage, not only reference notes',()=>{
  const html=n.render('chain-rule',n.reduce('chain-rule',n.initial(),'weight','0'));
  assert.match(html,/nn-path-factor is-blocked/);
  assert.match(html,/One zero factor stops this path/);
});
test('Backprop does not claim a parameter update before a new forward pass',()=>{
  let s=n.initial();s=n.reduce('backprop',s,'step',2);
  assert.match(n.render('backprop',s),/Update prepared/);
  s=n.reduce('backprop',s,'step',3);
  assert.match(n.render('backprop',s),/Pixels and label stayed fixed/);
  assert.deepEqual(s.parameters,n.initialParameters());
});
