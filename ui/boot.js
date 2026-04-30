// External boot file so production CSP (`script-src 'self'`) can start the app.
// Keep the cache buster here rather than inline in index.html.
function showBootFailure(message) {
  const shell = document.getElementById('authShell');
  if (!shell) return;
  shell.classList.add('open');
  shell.setAttribute('aria-hidden', 'false');
  const title = document.getElementById('authTitle');
  const sub = document.getElementById('authSub');
  const err = document.getElementById('authError');
  if (title) title.textContent = 'Could not start';
  if (sub) sub.textContent = 'Refresh once the pod is ready.';
  if (err) err.textContent = message;
}

import(`./app.js?v=${Date.now()}`).catch((err) => {
  console.error('App module failed to load:', err);
  showBootFailure('The app script failed to load. Refresh once the pod is ready.');
});
