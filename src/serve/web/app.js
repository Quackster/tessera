// tessera web chat: sessions over /api/sessions, turns over SSE.
// No dependencies; same-origin fetch (no CORS involved).

let sessions = [];
let currentId = null;
let currentMessages = [];
// Live turns: at most one per session, keyed by session id. A turn owns
// that session's stream from the moment send is claimed until it ends.
// Only the turn whose session is active paints the shared message view,
// so a background stream never overwrites the chat on screen. The server
// queues device work, so several sessions may hold a live turn at once.
let live = {};
// True while a first message is creating its session, before that
// chat's live turn exists: a second click must not create a second one.
let creatingSession = false;
// Browser-side chat cache (localStorage): instant paint on reload.
// The server stays authoritative: a miss, or an id the server no
// longer knows, falls back to fetching (the prefill path).
let cachedFull = {};
let cacheWriteFailed = false;
// Server readiness: the HTTP API answers before the model finishes
// loading, so the composer stays disabled until /health is ok.
let serverReady = false;
// The reasoning-effort levels the model template offers (from /props);
// empty hides the difficulty selector.
let reasoningEfforts = [];

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
    const src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(code);
    return head + '<img class="svg-embed" src="' + src + '" alt="SVG">';
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

function renderMessages(messages, liveThinkingOpen, scroll, resetOpen) {
  const box = $('messages');
  // Preserve thinking boxes across re-renders: rebuilding the list
  // would force every box back to the streamed default, so a box the
  // user closed would snap open on the next token. A session switch
  // resets them: open state belongs to one chat, never the next.
  const openStates = resetOpen ? [] :
      [...box.querySelectorAll('details.thinking')]
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
    const busy = session.busy || !!live[session.id];
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = (busy ? '\u25cf ' : '') +
        session.message_count + ' messages';
    if (busy) meta.classList.add('busy-dot');
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
          updateControls();
        }
      } catch (error) { showError(String(error)); }
    });
    item.append(title, meta, del);
    item.addEventListener('click', () => openSession(session.id));
    list.appendChild(item);
  }
}

// Apply a title the server generated before the answer streamed, so
// the sidebar names the chat while its reply is still generating. The
// list entry and the cached chat both update, whether or not this chat
// is the one on screen.
function applyTitle(id, title) {
  if (!title) return;
  if (cachedFull[id]) cachedFull[id].title = title;
  const session = sessions.find((entry) => entry.id === id);
  if (session) session.title = title;
  writeCache();
  renderSidebar();
}

// The live turn of the open chat, or null when that chat is idle.
function activeTurnState() {
  return currentId ? live[currentId] : null;
}

// A live turn as the trailing assistant message the view renders.
function liveMessage(turn) {
  return {
    role: 'assistant',
    content: turn.content,
    reasoning_content: turn.thinking,
    stopped: false,
    created_ms: turn.started,
    tokens_per_second: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    prefill: (!turn.thinking && !turn.content) ? turn.prefill : '',
  };
}

function statusForTurn(turn) {
  if (turn.paused) return 'Paused';
  if (turn.generating) return 'Generating\u2026';
  if (turn.prefill) return turn.prefill;
  return 'Thinking\u2026';
}

// Repaint the open chat from a live turn. A turn for another chat is
// left in `live` untouched: it becomes visible when that chat opens.
function drawTurn(sid, resetOpen) {
  if (sid !== currentId) return;
  const turn = live[sid];
  if (!turn) return;
  renderMessages(turn.base.concat([liveMessage(turn)]),
      turn.thinkingOpen, 'sticky', resetOpen);
  setStatus(statusForTurn(turn));
}

// Controls act on the open chat: send and retry are blocked only while
// that chat itself generates, so another chat may still request a queued
// turn. Stop, pause and resume show only for the open chat's live turn.
function updateControls() {
  const turn = activeTurnState();
  $('send').disabled = !serverReady || !!turn || creatingSession;
  $('retry').disabled = !serverReady || !currentId || !!turn;
  $('new-chat').disabled = !serverReady;
  $('input').disabled = !serverReady;
  $('stop').hidden = !turn;
  $('pause').hidden = !turn || turn.paused;
  $('resume').hidden = !turn || !turn.paused;
  // No open chat shows the centered start button in place of the view.
  const empty = !currentId;
  $('empty-state').hidden = !empty;
  $('messages').hidden = empty;
  $('composer').hidden = empty;
  $('start-chat').disabled = !serverReady;
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
  // No auto-select: an id the server dropped clears the view, and the
  // initial load opens no chat (the center button starts one).
  if (currentId && !sessions.some((s) => s.id === currentId)) {
    currentId = null;
  }
  writeCache();
  renderSidebar();
  updateControls();
}

async function openSession(id) {
  currentId = id;
  // A live turn owns the chat until it ends: paint its in-progress state
  // instead of the stored history, which lacks the streaming assistant.
  const turn = live[id];
  if (turn) {
    currentMessages = turn.base;
    drawTurn(id, true);
    renderSidebar();
    updateControls();
    return;
  }
  const session = await api('GET', '/api/sessions/' + id);
  if (currentId !== id) return;  // a newer selection won the race
  currentMessages = session.messages.map(normalizeMessage);
  cachedFull[session.id] = {
    id: session.id,
    title: session.title,
    messages: currentMessages,
  };
  writeCache();
  renderMessages(currentMessages, false, 'bottom', true);
  renderSidebar();
  updateControls();
  setStatus(session.busy ? 'Generating\u2026' : 'Ready');
}

// Stream one turn (chat or retry) over `base` history into session `sid`:
// the user bubble and an empty assistant show instantly when that session
// is active, then live deltas fill in. The turn lives in `live[sid]` for
// its whole life, so the stream keeps running while another chat is open
// and only paints the view when `sid` is the active chat.
async function streamTurn(path, payload, base, sid) {
  const turn = {
    base,
    thinking: '',
    content: '',
    prefill: '',
    generating: false,
    paused: false,
    thinkingOpen: true,
    started: Date.now(),
    controller: new AbortController(),
  };
  live[sid] = turn;
  if (sid === currentId) showError('');
  updateControls();
  renderSidebar();
  drawTurn(sid);
  const key = apiKey();
  const headers = {'Content-Type': 'application/json'};
  if (key) headers['x-api-key'] = key;
  try {
    const response = await fetch(path, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: turn.controller.signal,
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
          if (delta.session_title) {
            applyTitle(sid, delta.session_title);
            continue;
          }
          if (delta.prefill) {
            const done = Number(delta.prefill.done) || 0;
            const total = Number(delta.prefill.total) || 0;
            turn.prefill = total > 0
                ? 'Prefill ' + done + '/' + total + ' (' +
                  Math.round(done / total * 100) + '%)'
                : 'Prefill ' + done + ' tokens';
            drawTurn(sid);
            continue;
          }
          turn.generating = true;
          if (delta.reasoning_content) turn.thinking += delta.reasoning_content;
          if (delta.content) turn.content += delta.content;
          turn.thinkingOpen = !turn.content;
          drawTurn(sid);
        }
      }
    }
  } catch (error) {
    if (error.name !== 'AbortError' && sid === currentId) {
      showError(String(error));
    }
  } finally {
    if (live[sid] !== turn) return;  // superseded by a newer turn
    delete live[sid];
    // Authoritative state (stopped flags, usage) comes from the server.
    try {
      const session = await api('GET', '/api/sessions/' + sid);
      cachedFull[sid] = {
        id: sid,
        title: session.title,
        messages: session.messages.map(normalizeMessage),
      };
      writeCache();
    } catch (error) {
      if (sid === currentId) showError(String(error));
    }
    await refreshSessions();
    if (sid === currentId && cachedFull[sid]) {
      currentMessages = cachedFull[sid].messages;
      renderMessages(currentMessages, false, 'sticky', true);
      setStatus('Ready');
    }
  }
}

async function sendMessage() {
  const input = $('input');
  const text = input.value.trim();
  if (!text || !serverReady) return;
  if (currentId && live[currentId]) return;  // this chat is generating
  let sid = currentId;
  // No session yet: open one first, the message is not lost.
  if (!sid) {
    if (creatingSession) return;
    creatingSession = true;
    updateControls();
    try {
      const created = await api('POST', '/api/sessions', {});
      sid = created.id;
      currentId = sid;
      currentMessages = [];
      await refreshSessions();
    } catch (error) {
      showError(String(error));
      return;
    } finally {
      creatingSession = false;
      updateControls();
    }
    if (currentId !== sid) return;  // a newer selection won the race
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
  cachedFull[sid] = {
    id: sid,
    title: (cachedFull[sid] && cachedFull[sid].title) || 'New chat',
    messages: base,
  };
  writeCache();
  const payload = {
    message: text,
    stream: true,
    max_completion_tokens: completionBudget(),
    max_thinking_tokens: thinkingBudget(),
    enable_thinking: $('thinking').checked,
  };
  const effort = reasoningEffort();
  if (effort) payload.reasoning_effort = effort;
  await streamTurn('/api/sessions/' + sid + '/chat', payload, base, sid);
}

async function retryTurn() {
  if (!serverReady || !currentId || live[currentId]) return;
  // Drop trailing assistant turns locally, mirroring the server, so
  // the fresh answer streams after the right history.
  const base = currentMessages.slice();
  while (base.length && base[base.length - 1].role === 'assistant') {
    base.pop();
  }
  currentMessages = base;
  cachedFull[currentId] = {
    id: currentId,
    title: (cachedFull[currentId] && cachedFull[currentId].title) ||
        'New chat',
    messages: base,
  };
  writeCache();
  const payload = {
    stream: true,
    max_completion_tokens: completionBudget(),
    max_thinking_tokens: thinkingBudget(),
    enable_thinking: $('thinking').checked,
  };
  const effort = reasoningEffort();
  if (effort) payload.reasoning_effort = effort;
  await streamTurn('/api/sessions/' + currentId + '/retry', payload, base,
                   currentId);
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

// The selected reasoning effort, or '' to let the template use its own
// default (an empty value must be omitted from the request, not sent).
function reasoningEffort() {
  const select = $('reasoning-effort');
  return select ? select.value : '';
}

// Fill the difficulty selector from the levels the template validates
// (/props.reasoning_efforts) and keep it hidden when the model template
// has no reasoning_effort knob. The last choice is restored from
// localStorage.
function applyReasoningEfforts(efforts) {
  reasoningEfforts = Array.isArray(efforts) ? efforts : [];
  const row = $('effort-row');
  const select = $('reasoning-effort');
  if (!row || !select) return;
  select.innerHTML = '';
  for (const effort of reasoningEfforts) {
    const option = document.createElement('option');
    option.value = effort;
    option.textContent = effort;
    select.appendChild(option);
  }
  row.hidden = reasoningEfforts.length === 0;
  if (reasoningEfforts.length) {
    const saved = localStorage.getItem('tessera_effort');
    if (saved && reasoningEfforts.includes(saved)) select.value = saved;
    select.addEventListener('change', () => {
      localStorage.setItem('tessera_effort', select.value);
    });
  }
}

async function loadProps() {
  try {
    const props = await api('GET', '/props');
    if (props) applyReasoningEfforts(props.reasoning_efforts);
  } catch (error) {
    // The selector is optional; a props failure just hides it.
  }
}

// Controls act on the open chat's live turn.
async function stopTurn() {
  const turn = activeTurnState();
  if (!currentId) return;
  try { await api('POST', '/api/sessions/' + currentId + '/stop'); }
  catch (error) { showError(String(error)); }
  if (turn) turn.controller.abort();
}

async function pauseTurn() {
  const turn = activeTurnState();
  if (!currentId || !turn) return;
  try {
    await api('POST', '/api/sessions/' + currentId + '/pause');
    turn.paused = true;
    updateControls();
    setStatus(statusForTurn(turn));
  } catch (error) { showError(String(error)); }
}

async function resumeTurn() {
  const turn = activeTurnState();
  if (!currentId || !turn) return;
  try {
    await api('POST', '/api/sessions/' + currentId + '/resume');
    turn.paused = false;
    updateControls();
    setStatus(statusForTurn(turn));
  } catch (error) { showError(String(error)); }
}

async function newSession() {
  const created = await api('POST', '/api/sessions', {});
  await refreshSessions();
  await openSession(created.id);
}

// The center button on an empty view: open a fresh chat and focus it.
async function startChat() {
  await newSession();
  $('input').focus();
}

// Ask /health: 200 means the model is loaded and warm. While loading,
// show the loader's step and keep polling; a failed load is final.
async function checkHealth() {
  try {
    const response = await fetch('/health');
    if (response.ok) return true;
    let status = 'loading';
    let detail = '';
    try {
      const data = await response.json();
      status = data.status || 'loading';
      detail = data.detail || '';
    } catch (e) { /* plain text body */ }
    if (status === 'failed') {
      setStatus('Load failed');
      showError(detail || 'The model failed to load.');
      return 'failed';
    }
    setStatus(detail ? 'Loading: ' + detail + '\u2026' : 'Loading\u2026');
    return false;
  } catch (error) {
    setStatus('Connecting\u2026');
    return false;
  }
}

// Poll until the model is ready, then enable the composer and load the
// session list. A failed load stops the wait and shows the reason.
async function awaitServer() {
  for (;;) {
    const state = await checkHealth();
    if (state === true) {
      serverReady = true;
      updateControls();
      setStatus('Ready');
      try {
        await loadProps();
        await refreshSessions();
        if (currentId) await openSession(currentId);
      } catch (error) {
        showError(String(error));
      }
      return;
    }
    serverReady = false;
    updateControls();
    if (state === 'failed') return;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
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
  $('start-chat').addEventListener('click', () => {
    startChat().catch((error) => showError(String(error)));
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
  // Cache-aside boot: paint the left-hand chat list from the browser
  // cache instantly, then poll /health. No chat opens on load; the
  // center button starts one. The list refreshes from the server once
  // the model is ready.
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
    renderSidebar();
  }
  setStatus('Connecting\u2026');
  updateControls();
  awaitServer();
}

document.addEventListener('DOMContentLoaded', init);
