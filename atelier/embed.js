// embed.js — Standalone embed helper for daniellobato.me
//
// Drop a viz into any page with just:
//
//   <div class="atelier-embed" data-viz="attention"></div>
//   <script src="path/to/atelier-viz.iife.js"></script>
//   <script src="path/to/embed.js"></script>
//
// Or mount programmatically:
//
//   AtelierEmbed.render('attention', document.getElementById('my-div'));
//
// The embed wrapper creates the context stubs (takeaway/metrics) that
// the React viz expects, so you don't need studio.js or hybrid-renderers.

(function () {
  'use strict';

  function createContext(container) {
    // The viz write to these DOM nodes via setMetrics / setTakeaway.
    // We create them inside the container so they show up naturally.
    const takeaway = document.createElement('p');
    takeaway.className = 'atelier-embed-takeaway';
    takeaway.style.cssText =
      "margin:12px 0 0;font-family:'EB Garamond',Georgia,serif;font-style:italic;font-size:14px;color:#8a8680;line-height:1.6;";

    const metrics = document.createElement('div');
    metrics.className = 'atelier-embed-metrics';
    metrics.style.cssText =
      'display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;';

    return { takeaway, metrics };
  }

  function render(vizKey, container, opts) {
    if (!window.AtelierReactViz) {
      console.error(
        '[atelier-embed] AtelierReactViz not found. Load atelier-viz.iife.js before embed.js.',
      );
      return;
    }

    opts = opts || {};
    var state = opts.state || {};

    // Clear and build structure.
    container.innerHTML = '';
    container.style.cssText += ';max-width:860px;width:100%;';

    var stageRoot = document.createElement('div');
    stageRoot.className = 'atelier-embed-stage';
    container.appendChild(stageRoot);

    var ctx = createContext(container);

    // Only append takeaway/metrics if user hasn't opted out.
    if (opts.showTakeaway !== false) container.appendChild(ctx.takeaway);
    if (opts.showMetrics !== false) container.appendChild(ctx.metrics);

    window.AtelierReactViz.mount(vizKey, {
      stageRoot: stageRoot,
      section: { viz: vizKey },
      state: state,
      context: ctx,
    });

    return {
      update: function (newState) {
        window.AtelierReactViz.mount(vizKey, {
          stageRoot: stageRoot,
          section: { viz: vizKey },
          state: newState || state,
          context: ctx,
        });
      },
      unmount: function () {
        window.AtelierReactViz.unmount(stageRoot);
      },
    };
  }

  // Auto-scan: any element with class="atelier-embed" + data-viz="..." gets
  // a viz mounted into it on DOMContentLoaded.
  function autoMount() {
    var els = document.querySelectorAll('.atelier-embed[data-viz]');
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      var vizKey = el.getAttribute('data-viz');
      if (vizKey) render(vizKey, el);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoMount);
  } else {
    autoMount();
  }

  // Expose for programmatic use.
  window.AtelierEmbed = {
    render: render,
    vizKeys: window.AtelierReactViz ? window.AtelierReactViz.vizKeys : [],
  };
})();
