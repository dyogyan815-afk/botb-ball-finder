// ==UserScript==
// @name         Local Coordinate Research Panel
// @namespace    local-human-coordinate-emulator
// @version      0.1.0
// @description  Upload an image to a local prediction API. Does not click, submit, or automate BOTB.
// @match        https://www.botb.com/spot-the-ball*
// @grant        none
// ==/UserScript==
(function () {
  'use strict';
  const box = document.createElement('aside');
  box.style = 'position:fixed;z-index:2147483647;right:12px;bottom:12px;background:#111;color:#fff;padding:12px;border-radius:8px;font:14px system-ui;max-width:280px';
  box.innerHTML = '<b>Local research estimator</b><p style="margin:6px 0">Upload an image file. This panel never submits or clicks the competition.</p><input id="lc-file" type="file" accept="image/*"><pre id="lc-out" style="white-space:pre-wrap"></pre>';
  document.body.appendChild(box);
  box.querySelector('#lc-file').addEventListener('change', async (event) => {
    const file = event.target.files[0]; if (!file) return;
    const out = box.querySelector('#lc-out'); out.textContent = 'Analyzing locally…';
    const form = new FormData(); form.append('image', file);
    try {
      const response = await fetch('http://127.0.0.1:8000/predict', { method: 'POST', body: form });
      const data = await response.json();
      out.textContent = response.ok ? `x: ${data.x.toFixed(1)}\ny: ${data.y.toFixed(1)}\nuncertainty: ${data.uncertainty_px.toFixed(1)} px\nconfidence: ${data.confidence.toFixed(2)}\n\n${data.warning}` : (data.detail || 'Prediction failed');
    } catch (e) { out.textContent = 'Backend unavailable. Start uvicorn on port 8000.'; }
  });
})();
