/* In-browser Python exercises: a plain editor (no completion) that runs hidden tests with Pyodide.
   Usage: Atelier.mountExercise(element, { id, starter, tests, solution }) */
(function () {
  const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
  let pyPromise = null;

  const HARNESS = `
import json, sys, io, traceback, contextlib, warnings

def __atelier_run(user_src, test_src):
    ns = {"__name__": "solution"}
    out = io.StringIO()
    results = []
    try:
        with contextlib.redirect_stdout(out), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(compile(user_src, "<your code>", "exec"), ns)
    except Exception:
        tb = traceback.format_exc().splitlines()
        return json.dumps({"results": [{"name": "your code runs", "ok": False, "msg": "\\n".join(tb[-3:])}], "stdout": out.getvalue()})
    tns = dict(ns)
    exec(compile(test_src, "<tests>", "exec"), tns)
    for name, fn in list(tns.items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            with contextlib.redirect_stdout(out), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn()
            results.append({"name": name, "ok": True, "msg": ""})
        except AssertionError as e:
            results.append({"name": name, "ok": False, "msg": str(e) or "assertion failed"})
        except NotImplementedError:
            results.append({"name": name, "ok": False, "msg": "not implemented yet"})
        except Exception as e:
            results.append({"name": name, "ok": False, "msg": f"{type(e).__name__}: {e}"})
    return json.dumps({"results": results, "stdout": out.getvalue()})
`;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src; s.onload = resolve; s.onerror = () => reject(new Error('could not load ' + src));
      document.head.appendChild(s);
    });
  }

  function getPython(status) {
    if (!pyPromise) {
      pyPromise = (async () => {
        status('Loading Python (first run only, about 10 MB)…');
        await loadScript(PYODIDE + 'pyodide.js');
        const py = await window.loadPyodide({ indexURL: PYODIDE });
        status('Loading numpy…');
        await py.loadPackage('numpy');
        py.runPython(HARNESS);
        return py;
      })();
      pyPromise.catch(() => { pyPromise = null; });
    }
    return pyPromise;
  }

  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* ignore */ } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* ignore */ } },
  };

  function humanize(name) {
    return name.replace(/^test_/, '').replace(/_/g, ' ');
  }

  function wireEditor(ta, run) {
    ta.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); run(); return; }
      const { selectionStart: s, selectionEnd: t, value: v } = ta;
      if (e.key === 'Tab') {
        e.preventDefault();
        const lineStart = v.lastIndexOf('\n', s - 1) + 1;
        if (e.shiftKey) {
          const line = v.slice(lineStart);
          const n = (line.match(/^ {1,4}/) || [''])[0].length;
          ta.setRangeText('', lineStart, lineStart + n, 'end');
          ta.setSelectionRange(Math.max(lineStart, s - n), Math.max(lineStart, t - n));
        } else {
          ta.setRangeText('    ', s, t, 'end');
        }
      } else if (e.key === 'Enter' && !e.shiftKey) {
        const lineStart = v.lastIndexOf('\n', s - 1) + 1;
        const line = v.slice(lineStart, s);
        let indent = (line.match(/^ */) || [''])[0];
        if (/:\s*$/.test(line)) indent += '    ';
        e.preventDefault();
        ta.setRangeText('\n' + indent, s, t, 'end');
      }
    });
  }

  function mountExercise(root, cfg) {
    const key = 'atelier:ex:' + cfg.id;
    root.classList.add('ex');
    root.innerHTML = `
      <textarea spellcheck="false" autocapitalize="off" autocomplete="off" autocorrect="off" aria-label="Python editor"></textarea>
      <div class="btns">
        <button class="btn primary" data-run>Run tests</button>
        <button class="btn" data-reset>Reset to starter</button>
        <span class="status" aria-live="polite"></span>
      </div>
      <ul class="results"></ul>
      <details class="more"><summary>Reference solution (counts as assisted)</summary><pre class="solution"></pre></details>`;
    const ta = root.querySelector('textarea');
    const status = root.querySelector('.status');
    const results = root.querySelector('.results');
    root.querySelector('.solution').textContent = cfg.solution.trim() + '\n';
    ta.value = store.get(key) || cfg.starter.trim() + '\n';
    ta.addEventListener('input', () => store.set(key, ta.value));

    const setStatus = (t) => { status.textContent = t; };
    const runBtn = root.querySelector('[data-run]');
    async function run() {
      runBtn.disabled = true; results.innerHTML = '';
      try {
        const py = await getPython(setStatus);
        setStatus('Running…');
        const fn = py.globals.get('__atelier_run');
        const res = JSON.parse(fn(ta.value, cfg.tests));
        fn.destroy();
        const passed = res.results.filter((r) => r.ok).length;
        setStatus(`${passed} / ${res.results.length} passed`);
        for (const r of res.results) {
          const li = document.createElement('li');
          li.className = r.ok ? 'pass' : 'fail';
          li.textContent = (r.ok ? '✓ ' : '✗ ') + humanize(r.name) + (r.msg ? ' — ' + r.msg : '');
          results.appendChild(li);
        }
        if (res.stdout) {
          const li = document.createElement('li');
          li.textContent = 'stdout:\n' + res.stdout.trimEnd();
          results.appendChild(li);
        }
      } catch (err) {
        setStatus('');
        const li = document.createElement('li');
        li.className = 'err'; li.textContent = String(err);
        results.appendChild(li);
      } finally {
        runBtn.disabled = false;
      }
    }
    runBtn.addEventListener('click', run);
    root.querySelector('[data-reset]').addEventListener('click', () => {
      if (!confirm('Replace your code with the starter?')) return;
      ta.value = cfg.starter.trim() + '\n'; store.del(key); results.innerHTML = ''; setStatus('');
    });
    wireEditor(ta, run);
  }

  window.Atelier = Object.assign(window.Atelier || {}, { mountExercise, store });
})();
