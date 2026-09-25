/* Shared lifecycle and interaction handling; each lesson owns its mechanism. */
(() => {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

  function createModule(spec) {
    const kinds = Object.keys(spec.content);
    const states = new Map();
    const roots = new Map();
    const bindings = new WeakMap();
    const stateFor = kind => {
      if (!states.has(kind)) states.set(kind, spec.initial(kind));
      return states.get(kind);
    };
    function paint(kind, root) {
      const detailKey = (node,index) => node.dataset.detailKey || `${index}:${node.querySelector('summary')?.textContent}`;
      const openedDetails = new Set([...root.querySelectorAll('details')].flatMap((node,index) => node.open ? [detailKey(node,index)] : []));
      const focused = root.contains(document.activeElement) ? document.activeElement : null;
      const action = focused?.dataset.action;
      const value = focused?.dataset.value;
      const selection = focused && typeof focused.selectionStart === 'number' ? [focused.selectionStart, focused.selectionEnd] : null;
      root.innerHTML = `<section class="lesson ml-lab" data-family="${escape(spec.id)}">${spec.render(kind, stateFor(kind))}</section>`;
      [...root.querySelectorAll('details')].forEach((detail,index) => {
        if (openedDetails.has(detailKey(detail,index))) detail.open = true;
      });
      if (action) {
        const replacement = [...root.querySelectorAll('[data-action]')].find(node => node.dataset.action === action && node.dataset.value === value);
        if (replacement && !replacement.disabled) {
          replacement.focus({preventScroll:true});
          if (selection && replacement.setSelectionRange) replacement.setSelectionRange(...selection);
        }
      }
    }
    function dispatch(kind, action, value) {
      const previous = stateFor(kind);
      const next = spec.reduce(kind, previous, action, value);
      if (next == null || next === previous || JSON.stringify(next) === JSON.stringify(previous)) return;
      states.set(kind, next);
      for (const [root, mountedKind] of roots) {
        if (!root.isConnected) { unmount(root); continue; }
        if (mountedKind === kind) paint(kind, root);
      }
    }
    function render(kind, root) {
      if (!kinds.includes(kind)) throw new RangeError(`Unknown ${spec.id} lesson: ${kind}`);
      roots.set(root, kind);
      if (!bindings.has(root)) {
        const click = event => {
          const target = event.target.closest('[data-action]');
          if (!target || !root.contains(target) || target.disabled || /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;
          event.stopPropagation();
          if (target.tagName === 'SUMMARY') {
            // Controlled disclosures must capture the requested, not previous,
            // open state before the lesson replaces its markup.
            event.preventDefault();
            target.parentElement.open = !target.parentElement.open;
          }
          dispatch(roots.get(root), target.dataset.action, target.dataset.value);
        };
        const input = event => {
          const target = event.target.closest('input[data-action], select[data-action], textarea[data-action]');
          if (!target) return;
          event.stopPropagation();
          dispatch(roots.get(root), target.dataset.action, target.type === 'checkbox' ? target.checked : target.value);
        };
        const keydown = event => {
          const target = event.target.closest('[data-action][role="button"]');
          if (!target || !['Enter',' '].includes(event.key)) return;
          event.preventDefault(); event.stopPropagation();
          dispatch(roots.get(root), target.dataset.action, target.dataset.value);
        };
        const hover = event => {
          const target = event.target.closest('[data-action][data-hover="true"]');
          if (!target || target.contains(event.relatedTarget)) return;
          dispatch(roots.get(root), target.dataset.action, target.dataset.value);
        };
        root.addEventListener('click', click);
        root.addEventListener('input', input);
        root.addEventListener('keydown', keydown);
        root.addEventListener('mouseover', hover);
        bindings.set(root, {click,input,keydown,mouseover:hover});
      }
      paint(kind, root);
    }
    function unmount(root) {
      const handlers = bindings.get(root);
      if (handlers) for (const [event, handler] of Object.entries(handlers)) root.removeEventListener(event, handler);
      bindings.delete(root);
      roots.delete(root);
    }
    const api = {has:kind => kinds.includes(kind), content:spec.content, render, unmount};
    if (typeof window !== 'undefined') (window.AtelierLessonModules ||= []).push(api);
    return api;
  }
  const api = {createModule, escape};
  if (typeof window !== 'undefined') window.AtelierLab = api;
  if (typeof module !== 'undefined') module.exports = api;
})();
