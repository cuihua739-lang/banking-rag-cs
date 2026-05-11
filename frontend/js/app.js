/** Bank Intelligent Customer Service — Chat App */

const API = '/api';

// ===== DOM references =====
const $ = (sel) => document.querySelector(sel);
const chatMain = $('#chatMain');
const welcomeScreen = $('#welcomeScreen');
const messagesContainer = $('#messagesContainer');
const messageInput = $('#messageInput');
const btnSend = $('#btnSend');
const btnClear = $('#btnClear');
const statusIndicator = $('#statusIndicator');
const statusText = $('#statusText');
const statusInfo = $('#statusInfo');
const quickBtns = document.querySelectorAll('.quick-btn');

// ===== State =====
let messages = [];
let isProcessing = false;

// ===== Init =====
function init() {
  // Auto-resize textarea
  messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
  });

  // Send on Ctrl+Enter
  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      sendMessage();
    }
  });

  btnSend.addEventListener('click', sendMessage);
  btnClear.addEventListener('click', clearChat);

  // Quick question buttons
  quickBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      messageInput.value = btn.textContent;
      sendMessage();
    });
  });
}

// ===== Send message =====
async function sendMessage() {
  const query = messageInput.value.trim();
  if (!query || isProcessing) return;

  messageInput.value = '';
  messageInput.style.height = 'auto';
  isProcessing = true;
  btnSend.disabled = true;
  setStatus('active', '处理中...');

  // Hide welcome, show messages
  welcomeScreen.style.display = 'none';
  messagesContainer.style.display = 'flex';

  // Add user message
  addMessage('user', query);
  scrollToBottom();

  // Add assistant placeholder with typing indicator
  const assistMsg = addMessage('assistant', '', true);

  try {
    await streamChat(query, assistMsg);
    setStatus('ok', '就绪');
  } catch (err) {
    assistMsg.querySelector('.message-bubble').innerHTML = '抱歉，系统暂时出现异常，请您稍后再试或拨打客服热线 955XX。';
    setStatus('error', '请求失败');
    console.error('Chat error:', err);
  } finally {
    isProcessing = false;
    btnSend.disabled = false;
    messageInput.focus();
  }
}

// ===== SSE streaming =====
async function streamChat(query, msgEl) {
  const resp = await fetch(API + '/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });

  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalData = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const data = JSON.parse(line.slice(6));
        handleSSEEvent(data, query, msgEl);
        if (data.step === 'done') finalData = data;
      } catch (e) { /* skip parse errors */ }
    }
  }

  // Render final result
  if (finalData) {
    renderFinalAnswer(msgEl, finalData, query);
  }
}

function handleSSEEvent(data, query, msgEl) {
  const bubble = msgEl.querySelector('.message-bubble');

  switch (data.step) {
    case 'retrieving':
      statusText.textContent = '正在检索...';
      break;
    case 'retrieved':
      statusText.textContent = `检索到 ${data.candidates} 条相关知识`;
      break;
    case 'judging':
      statusText.textContent = '评估关联度...';
      break;
    case 'judged':
      statusText.textContent = `关联度评分: ${data.average_score}`;
      break;
    case 'expanding':
      statusText.textContent = '关联度不足，深度检索中...';
      bubble.innerHTML = '<p><em>正在扩大检索范围，寻找更相关的信息...</em></p>';
      break;
    case 'expanded':
      statusText.textContent = `查询改写为 ${data.variants} 个变体`;
      break;
    case 'retrieving_deep':
      statusText.textContent = '第二轮检索中...';
      break;
    case 'retrieved_deep':
      statusText.textContent = `深度检索到 ${data.candidates} 条知识`;
      break;
    case 'generating':
      statusText.textContent = '生成回答中...';
      bubble.innerHTML = '<p><em>正在整理回答...</em></p>';
      break;
  }
}

function renderFinalAnswer(msgEl, data, query) {
  const bubble = msgEl.querySelector('.message-bubble');

  // Render markdown-like answer
  let html = renderMarkdown(data.answer);

  // Add citations
  if (data.citations && data.citations.length > 0) {
    html += '<div class="citations-box"><details><summary>参考来源 (' + data.citations.length + '条)</summary>';
    data.citations.forEach((cite, i) => {
      html += '<div class="citation-item">';
      html += '<div class="cite-title">[' + (i + 1) + '] ' + escapeHtml(cite.doc_title || cite.section_title || '知识条目') + '</div>';
      if (cite.excerpt) {
        html += '<div class="cite-excerpt">' + escapeHtml(cite.excerpt) + '</div>';
      }
      html += '</div>';
    });
    html += '</details></div>';
  }

  // Add round info
  if (data.round === 2) {
    html += '<p style="font-size:11px;color:#9ca3af;margin-top:8px;">&#8505; 此回答经过深度检索（Round 2）</p>';
  }

  bubble.innerHTML = html;
  scrollToBottom();

  // Load follow-up suggestions
  if (query) loadSuggestions(query);
}

// ===== Simple Markdown renderer =====
function renderMarkdown(text) {
  if (!text) return '';
  let html = escapeHtml(text);

  // Tables
  html = html.replace(/\|(.+)\|\n\|[-| :]+\|\n((?:\|.+\|\n?)*)/g, function(match, header, rows) {
    let tbl = '<table><thead><tr>';
    header.split('|').filter(c => c.trim()).forEach(c => tbl += '<th>' + c.trim() + '</th>');
    tbl += '</tr></thead><tbody>';
    rows.trim().split('\n').forEach(row => {
      tbl += '<tr>';
      row.split('|').filter(c => c.trim()).forEach(c => tbl += '<td>' + c.trim() + '</td>');
      tbl += '</tr>';
    });
    tbl += '</tbody></table>';
    return tbl;
  });

  // Headers
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
  html = html.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>');
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, function(m) {
    return m.includes('<ul>') ? m : '<ol>' + m + '</ol>';
  });

  // Paragraphs
  html = html.replace(/\n\n/g, '</p><p>');
  html = '<p>' + html + '</p>';

  // Citation references
  html = html.replace(/\[(\d+)\]/g, '<sup class="citation-ref">[$1]</sup>');

  // Clean empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, '');
  html = html.replace(/<p><ul>/g, '<ul>');
  html = html.replace(/<\/ul><\/p>/g, '</ul>');
  html = html.replace(/<p><ol>/g, '<ol>');
  html = html.replace(/<\/ol><\/p>/g, '</ol>');

  return html;
}

function escapeHtml(str) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  return str.replace(/[&<>"']/g, c => map[c]);
}

// ===== Add message to UI =====
function addMessage(role, text, isStreaming) {
  const row = document.createElement('div');
  row.className = 'message-row ' + role;

  // Avatar
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = role === 'user' ? '👤' : '🏦';
  row.appendChild(avatar);

  // Bubble
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  if (role === 'user') {
    bubble.textContent = text;
  } else if (isStreaming) {
    bubble.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  } else {
    bubble.innerHTML = renderMarkdown(text);
  }
  row.appendChild(bubble);

  messagesContainer.appendChild(row);
  scrollToBottom();
  return row;
}

// ===== Helpers =====
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatMain.scrollTop = chatMain.scrollHeight;
  });
}

function setStatus(state, text, info) {
  statusIndicator.className = 'status-indicator ' + (state === 'active' ? 'active' : state === 'error' ? 'error' : '');
  statusText.textContent = text;
  statusInfo.textContent = info || '';
}

function clearChat() {
  if (isProcessing) return;
  messages = [];
  messagesContainer.innerHTML = '';
  messagesContainer.style.display = 'none';
  welcomeScreen.style.display = '';
  setStatus('ok', '对话已清空');
}

// ===== Load suggestions after answer =====
async function loadSuggestions(query) {
  try {
    const resp = await fetch(API + '/chat/suggestions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    if (resp.ok) {
      const data = await resp.json();
      if (data.suggestions && data.suggestions.length > 0) {
        showSuggestionChips(data.suggestions);
      }
    }
  } catch (e) { /* ignore */ }
}

function showSuggestionChips(suggestions) {
  // Remove existing suggestion row
  const existing = messagesContainer.querySelector('.suggestion-row');
  if (existing) existing.remove();

  const row = document.createElement('div');
  row.className = 'message-row assistant suggestion-row';
  const avatar = document.createElement('div');
  avatar.className = 'message-avatar';
  avatar.textContent = '💡';
  row.appendChild(avatar);

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.style.background = 'transparent';
  bubble.style.padding = '8px 0';
  bubble.innerHTML = '<p style="margin-bottom:8px;color:#6b7280;font-size:13px;">您可能还想问：</p>';

  const chips = document.createElement('div');
  chips.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;';
  suggestions.forEach(s => {
    const chip = document.createElement('button');
    chip.className = 'quick-btn';
    chip.textContent = s;
    chip.addEventListener('click', () => {
      messageInput.value = s;
      sendMessage();
    });
    chips.appendChild(chip);
  });
  bubble.appendChild(chips);
  row.appendChild(bubble);
  messagesContainer.appendChild(row);
  scrollToBottom();
}

// ===== Bootstrap =====
init();
messageInput.focus();
