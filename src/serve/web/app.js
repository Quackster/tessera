// tessera web chat: sessions over /api/sessions, turns over SSE.
// No dependencies; same-origin fetch (no CORS involved).

let sessions = [];
let currentId = null;
let currentMessages = [];
let streaming = false;
let paused = false;
let readerAbort = null;
// Turn tokens: only the newest turn may draw or reset the controls,
// so a stale turn never paints over the current view.
let activeTurn = 0;
let streamingId = null;
// Browser-side chat cache (localStorage): instant paint on reload.
// The server stays authoritative: a miss, or an id the server no
// longer knows, falls back to fetching (the prefill path).
let cachedFull = {};
let cacheWriteFailed = false;

const $ = (id) => document.getElementById(id);

function apiKey() {
  return localStorage.getItem('tessera_key') || '';
}

async function api(method, path, body) {
  const headers = {'Content-Type': 'application/json'};
  const key = apiKey();
  if (key) headers['x-api-key'] = key;
  const response = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { /* plain text */ }
  if (!response.ok) {
    const message = (data && data.error) ? data.error : text || response.status;
    throw new Error(message);
  }
  return data;
}

function showError(message) {
  const box = $('error');
  box.textContent = message;
  box.hidden = !message;
}

function setStatus(text) {
  $('status').textContent = text;
}

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Inline SVG from a fenced block, scripts and handlers removed.
function sanitizeSvg(svg) {
  return svg.replace(/<script[\s\S]*?<\/script\s*>/gi, '')
      .replace(/\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, '');
}

// Tiny dependency-free syntax highlighter: one single-pass split on
// a capture group, odd parts classified. Keywords, strings, comments
// and numbers for a few language families; markup gets tag mode.
function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const HIGHLIGHT_SPECS = {
  clike: {
    words: 'as async await break case catch class const continue debugger ' +
        'default delete do double else enum export extends false finally ' +
        'float for from function if implements import in instanceof int ' +
        'interface let long new null of private protected public return ' +
        'short static string struct super switch this throw true try ' +
        'typeof var void while with yield',
    line: ['//'],
    block: [['/*', '*/']],
    quotes: ['"', "'"],
  },
  python: {
    words: 'False None True and as assert async await break case class ' +
        'continue def del elif else except finally for from global if ' +
        'import in is lambda match nonlocal not or pass raise return try ' +
        'while with yield',
    line: ['#'],
    quotes: ['"', "'"],
  },
  rust: {
    words: 'Self as async await break const continue crate dyn else enum ' +
        'extern false fn for if impl in let loop match mod move mut pub ' +
        'ref return self static struct super trait true type unsafe use ' +
        'where while',
    line: ['//'],
    block: [['/*', '*/']],
    quotes: ['"', "'"],
  },
  bash: {
    words: 'case declare do done echo elif else esac exit export fi for ' +
        'function if in local readonly return select test then true false ' +
        'until while',
    line: ['#'],
    quotes: ['"', "'"],
  },
  json: {words: 'true false null', quotes: ['"']},
};

const HIGHLIGHT_ALIAS = {
  c: 'clike', h: 'clike', 'c++': 'clike', cpp: 'clike', cc: 'clike',
  cxx: 'clike', hpp: 'clike', cs: 'clike', java: 'clike', go: 'clike',
  js: 'clike', ts: 'clike', typescript: 'clike', javascript: 'clike',
  py: 'python', rs: 'rust', sh: 'bash', shell: 'bash', zsh: 'bash',
  json: 'json', xml: 'markup', html: 'markup', svg: 'markup',
};

const highlightCache = {};

function highlightSplitter(spec) {
  const parts = [];
  for (const [open, close] of spec.block || []) {
    parts.push(escapeRegExp(open) + '[\\s\\S]*?' + escapeRegExp(close));
  }
  for (const marker of spec.line || []) {
    parts.push(escapeRegExp(marker) + '[^\\n]*');
  }
  for (const quote of spec.quotes || []) {
    const q = escapeRegExp(quote);
    parts.push(q + '(?:[^' + quote + '\\n\\\\]|\\\\.)*' + q);
  }
  parts.push('\\b\\d[\\w.]*\\b');
  const words = spec.words.trim().split(/\s+/).map(escapeRegExp).join('|');
  if (words) parts.push('\\b(?:' + words + ')\\b');
  return new RegExp('(' + parts.join('|') + ')');
}

function highlightClass(token, spec) {
  for (const marker of spec.line || []) {
    if (token.startsWith(marker)) return 'com';
  }
  for (const [open] of spec.block || []) {
    if (token.startsWith(open)) return 'com';
  }
  for (const quote of spec.quotes || []) {
    if (token[0] === quote) return 'str';
  }
  if (/^\d/.test(token)) return 'num';
  return 'kw';
}

function highlightMarkup(code) {
  const splitter =
      /(<!--[\s\S]*?-->|<\/?[a-zA-Z][^<>\s/]*(?=\s|\/?>)|\/?>|"(?:[^"\\\n]|\\.)*"|'[^'\n]*'|[a-zA-Z-]+(?==))/;
  const parts = code.split(splitter);
  let out = '';
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 0 || parts[i] === undefined) {
      out += escapeHtml(parts[i] || '');
      continue;
    }
    const token = parts[i];
    let cls = 'attr';
    if (token.startsWith('<!--')) cls = 'com';
    else if (token[0] === '<' || token === '/' || token === '>' ||
        token === '/>') cls = 'tag';
    else if (token[0] === '"' || token[0] === "'") cls = 'str';
    out += '<span class="tok-' + cls + '">' + escapeHtml(token) + '</span>';
  }
  return out;
}

function highlightCode(code, lang) {
  const key = (lang || '').toLowerCase();
  const canonical = HIGHLIGHT_ALIAS[key] || key;
  if (canonical === 'markup') return highlightMarkup(code);
  const spec = HIGHLIGHT_SPECS[canonical];
  if (!spec) return escapeHtml(code);
  let splitter = highlightCache[canonical];
  if (!splitter) {
    splitter = highlightSplitter(spec);
    highlightCache[canonical] = splitter;
  }
  const parts = code.split(splitter);
  let out = '';
  for (let i = 0; i < parts.length; i++) {
    out += (i % 2 === 1 && parts[i] !== undefined)
        ? '<span class="tok-' + highlightClass(parts[i], spec) + '">' +
          escapeHtml(parts[i]) + '</span>'
        : escapeHtml(parts[i] || '');
  }
  return out;
}

function renderFence(info, code) {
  const cut = info.search(/[:\s]/);
  const lang = (cut < 0 ? info : info.slice(0, cut)).toLowerCase();
  const filename = (cut < 0 ? '' : info.slice(cut + 1)).trim();
  const head = filename
      ? '<div class="file-name">' + escapeHtml(filename) + '</div>' : '';
  if (lang === 'svg') {
    return head + '<div class="rendered">' + sanitizeSvg(code) + '</div>';
  }
  if (lang === 'html') {
    const src = code.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
    return head + '<iframe class="html-frame" sandbox="allow-scripts" srcdoc="' +
        src + '"></iframe>';
  }
  return head + '<pre><code>' + highlightCode(code, lang) + '</code></pre>';
}

function renderInline(text) {
  let out = escapeHtml(text);
  out = out.replace(/!\[([^\]]*)\]\((https?:\/\/[^\s)]+|data:image\/[^;\s]+;base64,[^\s)]+)\)/g,
      '<img alt="$1" src="$2" loading="lazy">');
  out = out.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  return out;
}

// Message text to HTML: fenced svg/html/code blocks render, markdown
// images embed, the rest is escaped paragraphs.
function renderMarkdown(text) {
  const fences = [];
  const fenced = text.replace(/```([^\n]*)\n([\s\S]*?)(?:```|$)/g,
      (match, info, code) => {
        fences.push(renderFence((info || '').trim(),
            code.replace(/\n$/, '')));
        return '\u0000' + (fences.length - 1) + '\u0000';
      });
  const paras = fenced.split(/\n\n+/).map((para) => {
    if (/^\u0000\d+\u0000$/.test(para.trim())) {
      return fences[Number(para.trim().slice(1, -1))];
    }
    const media = renderBareMedia(para.trim());
    if (media) return media;
    return '<p>' + renderInline(para).replace(/\n/g, '<br>') + '</p>';
  });
  return paras.join('\n');
}

// A bare image/video/audio URL on its own line embeds directly.
function renderBareMedia(line) {
  const url = line.match(/^(https?:\/\/[^\s)]+|data:image\/[^;\s]+;base64,[^\s)]+)$/);
  if (!url) return '';
  const src = escapeHtml(url[1]);
  if (/^data:image\//.test(url[1]) ||
      /\.(png|jpe?g|gif|webp|svg)(\?\S*)?$/i.test(url[1])) {
    return '<img src="' + src + '" loading="lazy" alt="">';
  }
  if (/\.(mp4|webm)(\?\S*)?$/i.test(url[1])) {
    return '<video controls preload="none" src="' + src + '"></video>';
  }
  if (/\.(mp3|wav|ogg)(\?\S*)?$/i.test(url[1])) {
    return '<audio controls preload="none" src="' + src + '"></audio>';
  }
  return '';
}

function messageHtml(message, thinkingOpen) {
  const prefill = (!message.reasoning_content && !message.content &&
      message.prefill)
      ? '<div class="prefill">' + escapeHtml(message.prefill) + '</div>'
      : '';
  const thinking = message.reasoning_content
      ? '<details class="thinking"' + (thinkingOpen ? ' open' : '') +
        '><summary>Thinking</summary><div class="thinking-body">' +
        renderMarkdown(message.reasoning_content) + '</div></details>'
      : '';
  const stopped = message.stopped
      ? ' <span class="stopped-badge">(stopped)</span>' : '';
  return messageMeta(message) +
      '<div class="content"><div class="role">' + escapeHtml(message.role) +
      stopped + '</div><div class="body">' + prefill + thinking +
      renderMarkdown(message.content || '') + '</div></div>';
}

// Left-hand per-message meta: the store timestamp plus, for assistant
// turns that recorded stats, the generation speed (token counts ride
// along as a hover title).
function messageMeta(message) {
  const time = message.created_ms
      ? '<div>' + escapeHtml(new Date(message.created_ms)
          .toLocaleTimeString()) + '</div>'
      : '';
  const speed = (message.role === 'assistant' &&
      message.tokens_per_second > 0)
      ? '<div>' + message.tokens_per_second.toFixed(1) + ' tok/s</div>'
      : '';
  const counts = (message.prompt_tokens > 0 || message.completion_tokens > 0)
      ? ' title="' + (message.prompt_tokens || 0) + ' prompt / ' +
        (message.completion_tokens || 0) + ' completion tokens"'
      : '';
  return '<div class="meta"' + counts + '>' + time + speed + '</div>';
}

function renderMessages(messages, liveThinkingOpen, scroll) {
  const box = $('messages');
  // Preserve thinking boxes across re-renders: rebuilding the list
  // would force every box back to the streamed default, so a box the
  // user closed would snap open on the next token.
  const openStates = [...box.querySelectorAll('details.thinking')]
      .map((details) => details.open);
  // Sticky scroll: follow the stream only while the view sits at the
  // bottom, so reading back never yanks. A fresh session opens at the
  // bottom instead.
  const sticky = scroll !== 'bottom' &&
      (box.scrollHeight - box.scrollTop - box.clientHeight < 80);
  box.innerHTML = messages.map((message) =>
      '<div class="message' + (message.stopped ? ' stopped' : '') + '">' +
      messageHtml(message, liveThinkingOpen) + '</div>').join('');
  box.querySelectorAll('details.thinking').forEach((details, index) => {
    if (index < openStates.length) details.open = openStates[index];
  });
  if (scroll === 'bottom' || sticky) box.scrollTop = box.scrollHeight;
}

function renderSidebar() {
  const list = $('sessions');
  list.innerHTML = '';
  for (const session of sessions) {
    const item = document.createElement('li');
    if (session.id === currentId) item.className = 'active';
    const title = document.createElement('span');
    title.className = 'title';
    title.textContent = session.title;
    title.title = 'Double-click to rename';
    title.addEventListener('dblclick', (event) => {
      event.stopPropagation();
      const input = document.createElement('input');
      input.value = session.title;
      input.addEventListener('keydown', async (keyEvent) => {
        if (keyEvent.key !== 'Enter') return;
        try {
          await api('PUT', '/api/sessions/' + session.id,
              {title: input.value.trim()});
          await refreshSessions();
        } catch (error) { showError(String(error)); }
      });
      title.replaceWith(input);
      input.focus();
    });
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = (session.busy ? '\u25cf ' : '') +
        session.message_count + ' messages';
    if (session.busy) meta.classList.add('busy-dot');
    const del = document.createElement('button');
    del.className = 'del';
    del.textContent = '\u00d7';
    del.title = 'Delete session';
    del.addEventListener('click', async (event) => {
      event.stopPropagation();
      try {
        await api('DELETE', '/api/sessions/' + session.id);
        if (session.id === currentId) currentId = null;
        await refreshSessions();
        if (currentId) await openSession(currentId);
        else {
          currentMessages = [];
          $('messages').innerHTML = '';
        }
      } catch (error) { showError(String(error)); }
    });
    item.append(title, meta, del);
    item.addEventListener('click', () => openSession(session.id));
    list.appendChild(item);
  }
}

function updateControls() {
  $('send').disabled = streaming;
  $('retry').disabled = streaming || !currentId;
  $('stop').hidden = !streaming;
  $('pause').hidden = !streaming || paused;
  $('resume').hidden = !streaming || !paused;
}

// One stored or cached message to the rendered shape; unknown fields
// default so older caches still paint.
function normalizeMessage(m) {
  return {
    role: m.role,
    content: m.content || '',
    reasoning_content: m.reasoning_content || '',
    stopped: !!m.stopped,
    created_ms: m.created_ms || 0,
    tokens_per_second: m.tokens_per_second || 0,
    prompt_tokens: m.prompt_tokens || 0,
    completion_tokens: m.completion_tokens || 0,
  };
}

// Read the browser chat cache; null when absent or malformed.
function readCache() {
  try {
    const raw = localStorage.getItem('tessera.sessions.v1');
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data || typeof data !== 'object' || !data.full ||
        typeof data.full !== 'object') {
      return null;
    }
    return data;
  } catch (e) {
    return null;
  }
}

// Persist the open chats to the browser cache. Quota or privacy mode
// drops the cache silently; the server stays the source of truth.
function writeCache() {
  if (cacheWriteFailed) return;
  try {
    localStorage.setItem('tessera.sessions.v1', JSON.stringify({
      currentId,
      full: cachedFull,
    }));
  } catch (e) {
    cacheWriteFailed = true;
    try {
      localStorage.removeItem('tessera.sessions.v1');
    } catch (ignored) { /* already giving up */ }
  }
}

async function refreshSessions() {
  sessions = (await api('GET', '/api/sessions')).sessions || [];
  const ids = new Set(sessions.map((s) => s.id));
  for (const id of Object.keys(cachedFull)) {
    if (!ids.has(id)) delete cachedFull[id];
  }
  for (const session of sessions) {
    if (cachedFull[session.id]) cachedFull[session.id].title = session.title;
  }
  if (!sessions.some((s) => s.id === currentId)) {
    currentId = sessions.length ? sessions[0].id : null;
  }
  writeCache();
  renderSidebar();
  updateControls();
}

async function openSession(id) {
  currentId = id;
  const session = await api('GET', '/api/sessions/' + id);
  currentMessages = session.messages.map(normalizeMessage);
  cachedFull[session.id] = {
    id: session.id,
    title: session.title,
    messages: currentMessages,
  };
  writeCache();
  renderMessages(currentMessages, false, 'bottom');
  renderSidebar();
  updateControls();
  setStatus(session.busy ? 'Generating\u2026' : 'Ready');
}

// Paint a cached chat without fetching (instant reload path); false
// on a miss, which falls back to the server fetch above.
function paintCached(id) {
  const cached = cachedFull[id];
  if (!cached || !Array.isArray(cached.messages)) {
    return false;
  }
  currentId = id;
  currentMessages = cached.messages.map(normalizeMessage);
  renderMessages(currentMessages, false, 'bottom');
  renderSidebar();
  updateControls();
  setStatus('Ready');
  return true;
}

// Stream one turn (chat or retry) over `base` history: the user bubble
// and an empty assistant show instantly, then live deltas fill in.
// Only this turn (by token) may draw or reset the controls.
async function streamTurn(path, payload, base, sid) {
  const myTurn = ++activeTurn;
  streamingId = sid;
  const key = apiKey();
  const headers = {'Content-Type': 'application/json'};
  if (key) headers['x-api-key'] = key;
  const controller = new AbortController();
  readerAbort = controller;
  streaming = true;
  paused = false;
  updateControls();
  setStatus('Thinking\u2026');
  showError('');
  let thinking = '';
  let content = '';
  let prefill = '';
  let generating = false;
  const turnStarted = Date.now();
  const draw = (thinkingOpen) => {
    if (myTurn !== activeTurn) return;
    renderMessages(base.concat([{
      role: 'assistant',
      content,
      reasoning_content: thinking,
      stopped: false,
      created_ms: turnStarted,
      tokens_per_second: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      prefill: (!thinking && !content) ? prefill : '',
    }]), thinkingOpen, 'sticky');
  };
  draw(true);
  try {
    const response = await fetch(path, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) {
      const text = await response.text();
      let message = text || String(response.status);
      try {
        const data = JSON.parse(text);
        if (data && data.error) message = data.error;
      } catch (e) { /* keep the raw text */ }
      throw new Error(message);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const events = buffer.split('\n\n');
      buffer = events.pop();
      for (const event of events) {
        for (const line of event.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') break;
          const chunk = JSON.parse(data);
          const delta = ((chunk.choices || [])[0] || {}).delta || {};
          if (delta.prefill) {
            const done = Number(delta.prefill.done) || 0;
            const total = Number(delta.prefill.total) || 0;
            prefill = total > 0
                ? 'Prefill ' + done + '/' + total + ' (' +
                  Math.round(done / total * 100) + '%)'
                : 'Prefill ' + done + ' tokens';
            setStatus(prefill);
            draw(true);
            continue;
          }
          if (!generating) {
            generating = true;
            setStatus('Generating\u2026');
          }
          if (delta.reasoning_content) thinking += delta.reasoning_content;
          if (delta.content) content += delta.content;
          draw(!content);
        }
      }
    }
  } catch (error) {
    if (error.name !== 'AbortError') showError(String(error));
  } finally {
    if (myTurn !== activeTurn) return;
    streaming = false;
    paused = false;
    readerAbort = null;
    streamingId = null;
    updateControls();
    // Authoritative state (stopped flags, usage) comes from the server.
    try {
      if (currentId) await openSession(currentId);
      setStatus('Ready');
    } catch (error) { showError(String(error)); }
  }
}

async function sendMessage() {
  const input = $('input');
  const text = input.value.trim();
  if (!text || streaming) return;
  // Claim the turn synchronously: a second Enter/click while the
  // session opens must not start a duplicate generation.
  streaming = true;
  updateControls();
  setStatus('Sending\u2026');
  // No session yet: open one first, the message is not lost.
  if (!currentId) {
    try {
      const created = await api('POST', '/api/sessions', {});
      currentId = created.id;
      await refreshSessions();
    } catch (error) {
      streaming = false;
      updateControls();
      setStatus('Ready');
      showError(String(error));
      return;
    }
  }
  input.value = '';
  const userMessage = {
    role: 'user',
    content: text,
    reasoning_content: '',
    stopped: false,
    created_ms: Date.now(),
    tokens_per_second: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
  };
  const base = currentMessages.concat([userMessage]);
  currentMessages = base;
  await streamTurn('/api/sessions/' + currentId + '/chat', {
    message: text,
    stream: true,
    max_completion_tokens: completionBudget(),
    max_thinking_tokens: thinkingBudget(),
    enable_thinking: $('thinking').checked,
  }, base, currentId);
}

async function retryTurn() {
  if (streaming || !currentId) return;
  // Drop trailing assistant turns locally, mirroring the server, so
  // the fresh answer streams after the right history.
  const base = currentMessages.slice();
  while (base.length && base[base.length - 1].role === 'assistant') {
    base.pop();
  }
  await streamTurn('/api/sessions/' + currentId + '/retry', {
    stream: true,
    max_completion_tokens: completionBudget(),
    max_thinking_tokens: thinkingBudget(),
    enable_thinking: $('thinking').checked,
  }, base, currentId);
}

// Budgets from the composer inputs: 0 (including a cleared field)
// means unlimited and fills the remaining context.
function completionBudget() {
  const input = $('max-completion-tokens');
  return Number(input.value) || Number(input.defaultValue);
}

function thinkingBudget() {
  const input = $('max-thinking-tokens');
  return Number(input.value) || 0;
}

// Controls target the live turn when one runs, else the open session.
function controlId() {
  return streamingId || currentId;
}

async function stopTurn() {
  const id = controlId();
  if (!id) return;
  try { await api('POST', '/api/sessions/' + id + '/stop'); }
  catch (error) { showError(String(error)); }
  if (readerAbort) readerAbort.abort();
}

async function pauseTurn() {
  const id = controlId();
  if (!id) return;
  try {
    await api('POST', '/api/sessions/' + id + '/pause');
    paused = true;
    updateControls();
    setStatus('Paused');
  } catch (error) { showError(String(error)); }
}

async function resumeTurn() {
  const id = controlId();
  if (!id) return;
  try {
    await api('POST', '/api/sessions/' + id + '/resume');
    paused = false;
    updateControls();
    setStatus('Generating\u2026');
  } catch (error) { showError(String(error)); }
}

async function newSession() {
  const created = await api('POST', '/api/sessions', {});
  await refreshSessions();
  await openSession(created.id);
}

function init() {
  // Script errors surface in the UI instead of silent buttons.
  window.addEventListener('error', (event) => {
    showError('UI error: ' + (event.message || 'unknown'));
  });
  $('api-key').value = apiKey();
  $('api-key').addEventListener('change', (event) => {
    localStorage.setItem('tessera_key', event.target.value);
  });
  $('new-chat').addEventListener('click', () => {
    newSession().catch((error) => showError(String(error)));
  });
  $('send').addEventListener('click', () => {
    sendMessage().catch((error) => showError(String(error)));
  });
  $('input').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage().catch((error) => showError(String(error)));
    }
  });
  $('stop').addEventListener('click', stopTurn);
  $('pause').addEventListener('click', pauseTurn);
  $('resume').addEventListener('click', resumeTurn);
  $('retry').addEventListener('click', () => {
    retryTurn().catch((error) => showError(String(error)));
  });
  // Cache-aside boot: paint the browser cache instantly, then refresh
  // from the server (a miss falls back to fetching). A dead server
  // keeps the cached view instead of an error page.
  const cached = readCache();
  if (cached) {
    cachedFull = cached.full;
    sessions = Object.values(cachedFull)
        .filter((s) => s && typeof s.id === 'string')
        .map((s) => ({
          id: s.id,
          title: s.title || 'New chat',
          message_count:
              Array.isArray(s.messages) ? s.messages.length : 0,
          busy: false,
        }));
    if (cached.currentId && cachedFull[cached.currentId]) {
      currentId = cached.currentId;
    } else if (sessions.length) {
      currentId = sessions[0].id;
    }
    renderSidebar();
    updateControls();
    if (currentId) paintCached(currentId);
  }
  refreshSessions()
      .then(() => { if (currentId) return openSession(currentId); })
      .catch((error) => {
        if (!sessions.length) showError(String(error));
        else setStatus('Offline: showing cached chats');
      });
}

document.addEventListener('DOMContentLoaded', init);
