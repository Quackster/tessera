// tessera web chat: sessions over /api/sessions, turns over SSE.
// No dependencies; same-origin fetch (no CORS involved).

let sessions = [];
let currentId = null;
let currentMessages = [];
let streaming = false;
let paused = false;
let readerAbort = null;

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

function renderFence(lang, code) {
  const name = lang.toLowerCase();
  if (name === 'svg') {
    return '<div class="rendered">' + sanitizeSvg(code) + '</div>';
  }
  if (name === 'html') {
    const src = code.replace(/&/g, '&amp;').replace(/"/g, '&quot;');
    return '<iframe class="html-frame" sandbox="allow-scripts" srcdoc="' +
        src + '"></iframe>';
  }
  return '<pre><code>' + escapeHtml(code) + '</code></pre>';
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
  const fenced = text.replace(/```(\w*)\n([\s\S]*?)(?:```|$)/g,
      (match, lang, code) => {
        fences.push(renderFence(lang || 'text', code.replace(/\n$/, '')));
        return '\u0000' + (fences.length - 1) + '\u0000';
      });
  const paras = fenced.split(/\n\n+/).map((para) => {
    if (/^\u0000\d+\u0000$/.test(para.trim())) {
      return fences[Number(para.trim().slice(1, -1))];
    }
    return '<p>' + renderInline(para).replace(/\n/g, '<br>') + '</p>';
  });
  return paras.join('\n');
}

function messageHtml(message, thinkingOpen) {
  const thinking = message.reasoning_content
      ? '<details class="thinking"' + (thinkingOpen ? ' open' : '') +
        '><summary>Thinking</summary>' +
        renderMarkdown(message.reasoning_content) + '</details>'
      : '';
  const stopped = message.stopped
      ? ' <span class="stopped-badge">(stopped)</span>' : '';
  return '<div class="role">' + escapeHtml(message.role) + stopped + '</div>' +
      '<div class="body">' + thinking +
      renderMarkdown(message.content || '') + '</div>';
}

function renderMessages(messages, liveThinkingOpen) {
  const box = $('messages');
  box.innerHTML = messages.map((message) =>
      '<div class="message' + (message.stopped ? ' stopped' : '') + '">' +
      messageHtml(message, liveThinkingOpen) + '</div>').join('');
  box.scrollTop = box.scrollHeight;
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

async function refreshSessions() {
  sessions = (await api('GET', '/api/sessions')).sessions || [];
  if (!sessions.some((s) => s.id === currentId)) {
    currentId = sessions.length ? sessions[0].id : null;
  }
  renderSidebar();
  updateControls();
}

async function openSession(id) {
  currentId = id;
  const session = await api('GET', '/api/sessions/' + id);
  currentMessages = session.messages.map((m) => ({
    role: m.role,
    content: m.content || '',
    reasoning_content: m.reasoning_content || '',
    stopped: !!m.stopped,
  }));
  renderMessages(currentMessages, false);
  renderSidebar();
  updateControls();
  setStatus(session.busy ? 'Generating\u2026' : 'Ready');
}

// Stream one turn (chat or retry) over `base` history: the user bubble
// and an empty assistant show instantly, then live deltas fill in.
async function streamTurn(path, payload, base) {
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
  const draw = (thinkingOpen) => {
    renderMessages(base.concat([{
      role: 'assistant',
      content,
      reasoning_content: thinking,
      stopped: false,
    }]), thinkingOpen);
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
          if (delta.reasoning_content) thinking += delta.reasoning_content;
          if (delta.content) content += delta.content;
          draw(!content);
        }
      }
    }
  } catch (error) {
    if (error.name !== 'AbortError') showError(String(error));
  } finally {
    streaming = false;
    paused = false;
    readerAbort = null;
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
  // No session yet: open one first, the message is not lost.
  if (!currentId) {
    try {
      const created = await api('POST', '/api/sessions', {});
      currentId = created.id;
      await refreshSessions();
    } catch (error) {
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
  };
  const base = currentMessages.concat([userMessage]);
  currentMessages = base;
  await streamTurn('/api/sessions/' + currentId + '/chat', {
    message: text,
    stream: true,
    max_tokens: Number($('max-tokens').value) ||
        Number($('max-tokens').defaultValue),
    enable_thinking: $('thinking').checked,
  }, base);
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
    max_tokens: Number($('max-tokens').value) ||
        Number($('max-tokens').defaultValue),
    enable_thinking: $('thinking').checked,
  }, base);
}

async function stopTurn() {
  if (!currentId) return;
  try { await api('POST', '/api/sessions/' + currentId + '/stop'); }
  catch (error) { showError(String(error)); }
  if (readerAbort) readerAbort.abort();
}

async function pauseTurn() {
  if (!currentId) return;
  try {
    await api('POST', '/api/sessions/' + currentId + '/pause');
    paused = true;
    updateControls();
    setStatus('Paused');
  } catch (error) { showError(String(error)); }
}

async function resumeTurn() {
  if (!currentId) return;
  try {
    await api('POST', '/api/sessions/' + currentId + '/resume');
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
  refreshSessions()
      .then(() => { if (currentId) return openSession(currentId); })
      .catch((error) => showError(String(error)));
}

document.addEventListener('DOMContentLoaded', init);
