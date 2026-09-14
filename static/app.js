"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

const els = {
  source: $("#cb-source"),
  index: $("#cb-index"),
  clear: $("#cb-clear"),
  export: $("#cb-export"),
  format: $("#cb-format"),
  status: $("#cb-status"),
  viewtabs: $("#cb-viewtabs"),
  viewChat: $("#cb-view-chat"),
  viewSearch: $("#cb-view-search"),
  viewGraph: $("#cb-view-graph"),
  searchInput: $("#cb-search-input"),
  searchResults: $("#cb-search-results"),
  messages: $("#cb-messages"),
  empty: $("#cb-empty"),
  quick: $("#cb-quick"),
  composer: $("#cb-composer"),
  input: $("#cb-input"),
  send: $("#cb-send"),
  newchat: $("#cb-newchat"),
  clearchat: $("#cb-clearchat"),
  gen: $("#cb-gen"),
  newsession: $("#cb-newsession"),
  graphEmbed: $("#cb-graph-embed"),
  graphBody: $("#cb-graph-body"),
  overlay: $("#cb-overlay"),
  overlayPath: $("#cb-overlay-path"),
  overlayAsk: $("#cb-overlay-ask"),
  overlayClose: $("#cb-overlay-close"),
  overlayBody: $("#cb-overlay-body"),
  footerInfo: $("#cb-footer-info"),
};

const state = { history: [], summary: "", streaming: false, graphHtml: "", files: [], sessions: {}, currentId: "" };

const LS_KEY = "codebase-qa.session.v4";

let uid = 0;
function newId() {
  return "s" + Date.now().toString(36) + "-" + String(uid++);
}

const AVATAR_USER =
  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="3.6"/><path d="M20 21a8 8 0 0 0-16 0"/></svg>';
const AVATAR_BOT =
  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8l4 2 4-2 4 2 4-2v8l-4-2-4 2-4-2-4 2Z"/><path d="M12 8V4"/><path d="M10 4h4"/></svg>';
const ICO_COPY =
  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H6V6a2 2 0 0 1 2-2h9"/></svg>';
const ICO_REGEN =
  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v6h6"/><path d="M3 9a9 9 0 0 1 15.4-3.3L21 8"/><path d="M21 21v-6h-6"/><path d="M21 15a9 9 0 0 1-15.4 3.3L3 16"/></svg>';

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function renderBlocks(t) {
  const out = [];
  let list = null;
  const closeList = () => {
    if (list) {
      out.push(list === "ul" ? "</ul>" : "</ol>");
      list = null;
    }
  };
  const openList = (ty) => {
    if (list === ty) return;
    closeList();
    list = ty;
    out.push(ty === "ul" ? "<ul>" : "<ol>");
  };
  t.split(/\n{2,}/).forEach((pg) => {
    if (!pg.trim()) return;
    const lines = pg.split("\n");
    const first = lines[0].trim();
    if (/^[-*_]{3,}$/.test(first)) {
      closeList();
      out.push("<hr>");
      return;
    }
    const h = first.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      closeList();
      out.push(`<h${h[1].length}>${inline(esc(h[2]))}</h${h[1].length}>`);
      return;
    }
    if (lines[0].startsWith("> ")) {
      closeList();
      out.push("<blockquote>" + lines.map((l) => inline(esc(l.replace(/^>\s?/, "")))).join("<br>") + "</blockquote>");
      return;
    }
    const lm = lines[0].match(/^\s*(?:[-*]|\d+\.)\s+(.*)$/);
    if (lm) {
      const ty = /^\d+\./.test(first) ? "ol" : "ul";
      openList(ty);
      out.push("<li>" + inline(esc(lm[1])) + "</li>");
      for (let i = 1; i < lines.length; i++) {
        const m = lines[i].match(/^\s*(?:[-*]|\d+\.)\s+(.*)$/);
        if (m) {
          const ty2 = /^\d+\./.test(lines[i].trim()) ? "ol" : "ul";
          if (ty2 !== ty) {
            closeList();
            openList(ty2);
          }
          out.push("<li>" + inline(esc(m[1])) + "</li>");
        } else {
          out[out.length - 1] += "<br>" + inline(esc(lines[i]));
        }
      }
      return;
    }
    closeList();
    out.push("<p>" + lines.map((l) => inline(esc(l))).join("<br>") + "</p>");
  });
  closeList();
  return out.join("");
}

function renderMD(text) {
  const out = [];
  const re = /```([\s\S]*?)```/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    out.push(renderBlocks(text.slice(last, m.index)));
    out.push("<pre><code>" + esc(m[1]) + "</code></pre>");
    last = m.index + m[0].length;
  }
  out.push(renderBlocks(text.slice(last)));
  return out.join("");
}

let stick = true;
function nearBottom() {
  return els.messages.scrollTop + els.messages.clientHeight >= els.messages.scrollHeight - 80;
}
function scrollBottom(force) {
  if (force || stick) els.messages.scrollTop = els.messages.scrollHeight;
}

function refreshEmpty() {
  els.empty.style.display = state.history.length ? "none" : "";
}

function refreshRegen() {
  const bots = $$(".msg.bot");
  bots.forEach((b, i) => {
    const r = b.querySelector('[data-act="regen"]');
    if (r) r.style.display = i === bots.length - 1 ? "" : "none";
  });
}

function addMessage(role, content) {
  const row = document.createElement("div");
  row.className = "msg " + role;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.innerHTML = role === "user" ? AVATAR_USER : AVATAR_BOT;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.dataset.raw = content || "";
  bubble.innerHTML = renderMD(content || "");
  row.append(avatar, bubble);
  if (role === "bot") {
    const act = document.createElement("div");
    act.className = "actions";
    act.innerHTML =
      '<button type="button" class="act" data-act="copy" title="Copy" aria-label="Copy">' +
      ICO_COPY +
      "</button>" +
      '<button type="button" class="act" data-act="regen" title="Regenerate" aria-label="Regenerate">' +
      ICO_REGEN +
      "</button>";
    row.appendChild(act);
  }
  els.messages.appendChild(row);
  if (role === "bot") activeMsg = row;
  linkFiles(bubble);
  refreshEmpty();
  refreshRegen();
  return bubble;
}

function setBubble(node, raw) {
  node.dataset.raw = raw;
  node.innerHTML = renderMD(raw);
  linkFiles(node);
}

function copyText(t, btn) {
  if (!t) return;
  const done = () => {
    if (btn) {
      btn.classList.add("ok");
      setTimeout(() => btn.classList.remove("ok"), 800);
    }
  };
  const fallback = () => {
    try {
      const ta = document.createElement("textarea");
      ta.value = t;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  };
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(t).then(done).catch(() => fallback() && done());
  } else if (fallback()) {
    done();
  }
}

/* ------------------------------------------------------------
   File linking, citations, steps, overlay, search, view switching
   ------------------------------------------------------------ */
let activeMsg = null;

function linkFiles(root) {
  const paths = (state.files || []).filter(Boolean).sort((a, b) => b.length - a.length);
  if (!paths.length) return;
  const re = new RegExp(
    "([^A-Za-z0-9_./\\\\-])(" +
      paths.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|") +
      ")(?=$|[^A-Za-z0-9_./\\\\-])",
    "g"
  );
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      if (n.parentElement.closest("pre,code,a,button,span.filelink")) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const nodes = [];
  let n;
  while ((n = walker.nextNode())) nodes.push(n);
  nodes.forEach((node) => {
    if (!node.nodeValue || !re.test(node.nodeValue)) return;
    re.lastIndex = 0;
    const frag = document.createDocumentFragment();
    let last = 0;
    let m;
    while ((m = re.exec(node.nodeValue))) {
      if (m.index > last) frag.appendChild(document.createTextNode(node.nodeValue.slice(last, m.index + m[1].length)));
      const span = document.createElement("span");
      span.className = "filelink";
      span.textContent = m[2];
      span.dataset.path = m[2];
      frag.appendChild(span);
      last = m.index + m[1].length + m[2].length;
    }
    if (!frag.childNodes.length) return;
    if (last < node.nodeValue.length) frag.appendChild(document.createTextNode(node.nodeValue.slice(last)));
    node.parentNode.replaceChild(frag, node);
  });
}

function renderSources(items) {
  const row = activeMsg;
  if (!row || !items || !items.length) return;
  let wrap = row.querySelector(".srcs");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "srcs";
    row.appendChild(wrap);
  }
  wrap.innerHTML = items
    .map(
      (s) =>
        '<button type="button" class="src" data-path="' +
        esc(s.source || "") +
        '" title="View ' +
        esc(s.source || "") +
        '">' +
        esc(s.source || "") +
        (s.start_line ? ":" + s.start_line : "") +
        "</button>"
    )
    .join("");
}

function addStep(text) {
  const row = activeMsg;
  if (!row) return;
  let wrap = row.querySelector(".steps");
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "steps";
    row.appendChild(wrap);
  }
  const chip = document.createElement("span");
  chip.className = "step";
  chip.textContent = "→ " + text;
  wrap.appendChild(chip);
  if (wrap.children.length > 6) wrap.removeChild(wrap.firstChild);
}

function closeOverlay() {
  els.overlay.hidden = true;
  els.overlayBody.innerHTML = "";
}

async function openFile(path) {
  els.overlayPath.textContent = path || "";
  els.overlayAsk.dataset.path = path || "";
  els.overlay.hidden = false;
  els.overlayBody.innerHTML = '<p class="hint">Loading…</p>';
  let data;
  try {
    const res = await fetch("/api/file?path=" + encodeURIComponent(path || ""));
    data = await res.json();
  } catch (e) {
    els.overlayBody.innerHTML = '<p class="hint">Error: ' + esc(e.message) + "</p>";
    return;
  }
  if (data.error) {
    els.overlayBody.innerHTML = '<p class="hint">' + esc(data.error) + "</p>";
    return;
  }
  const width = String(data.total_lines || 0).length;
  els.overlayBody.innerHTML = "";
  const pre = document.createElement("div");
  pre.className = "olines";
  let no = 0;
  (data.lines || []).forEach((line) => {
    no += 1;
    const row = document.createElement("div");
    row.className = "oline";
    const g = document.createElement("span");
    g.className = "ogutter";
    g.textContent = String(no).padStart(width);
    const t = document.createElement("span");
    t.className = "otext";
    t.textContent = line;
    row.append(g, t);
    pre.appendChild(row);
  });
  els.overlayBody.appendChild(pre);
  if (data.truncated) {
    const note = document.createElement("p");
    note.className = "hint onote";
    note.textContent = "File truncated — showing first " + data.lines.length + " of " + data.total_lines + " lines.";
    els.overlayBody.appendChild(note);
  }
  if (overlayPrevPath !== path) {
    overlayPrevPath = path;
    els.overlayBody.scrollTop = 0;
  }
}
let overlayPrevPath = null;

function runSearch(immediate) {
  const q = els.searchInput.value.trim();
  if (!runSearch._t) runSearch._t = null;
  if (!runSearch._seq) runSearch._seq = 0;
  clearTimeout(runSearch._t);
  if (!q) {
    renderSearchHint("Search the codebase for functions, symbols, error strings — results link straight to the source.");
    return;
  }
  const doSearch = async () => {
    const seq = ++runSearch._seq;
    els.searchResults.classList.add("loading");
    let data;
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, k: 10 }),
      });
      data = await res.json();
    } catch (e) {
      if (seq === runSearch._seq) renderSearchHint("Search failed: " + e.message);
      els.searchResults.classList.remove("loading");
      return;
    }
    els.searchResults.classList.remove("loading");
    if (seq !== runSearch._seq) return;
    renderResults(data.results || []);
  };
  if (immediate) doSearch();
  else runSearch._t = setTimeout(doSearch, 250);
}

function renderSearchHint(text) {
  els.searchResults.innerHTML = '<p class="hint">' + esc(text) + "</p>";
}

function renderResults(results) {
  if (!results.length) {
    renderSearchHint("No matches.");
    return;
  }
  els.searchResults.innerHTML = "";
  results.forEach((r) => {
    const card = document.createElement("article");
    card.className = "result";
    const loc = r.start_line ? "L" + r.start_line + (r.end_line && r.end_line !== r.start_line ? "-" + r.end_line : "") : "";
    card.innerHTML =
      '<div class="rhead"><span class="rpath">' +
      esc(r.source) +
      '</span><span class="rloc">' +
      esc(loc) +
      "</span></div>" +
      '<pre class="rsnip">' +
      esc(r.snippet) +
      "</pre>" +
      '<div class="ract">' +
      '<button class="act" data-a="open" data-path="' +
      esc(r.source) +
      '">Open</button>' +
      '<button class="act" data-a="ask" data-path="' +
      esc(r.source) +
      '">Ask</button></div>';
    els.searchResults.appendChild(card);
  });
}

function switchTo(view) {
  els.viewtabs.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x.dataset.view === view));
  const views = { chat: els.viewChat, search: els.viewSearch, graph: els.viewGraph };
  Object.entries(views).forEach(([name, el]) => el.classList.toggle("active", name === view));
  if (view === "graph") showGraph();
  if (view === "search") els.searchInput.focus();
}

function askAbout(path) {
  switchTo("chat");
  els.input.value = path ? "What does " + path + " do?" : "";
  autosize();
  els.input.focus();
}

const reducedMotion =
  window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

const typer = {
  bubble: null,
  target: "",
  shown: 0,
  final: false,
  timer: null,
  start(bubble) {
    this.stop();
    this.bubble = bubble;
    this.target = "";
    this.shown = 0;
    this.final = false;
    bubble.classList.add("streaming", "caret");
    if (reducedMotion) {
      this.flush();
    } else {
      this.timer = setInterval(() => this.tick(), 16);
    }
  },
  append(text) {
    if (!this.bubble) return;
    this.target += text;
    if (reducedMotion) {
      this.shown = this.target.length;
      this._render();
    }
  },
  finalize() {
    this.final = true;
    if (this.shown >= this.target.length) this.stop();
  },
  force(text) {
    if (!this.bubble) return;
    if (!this.target) this.target = text;
    this.flush();
  },
  finishNow() {
    if (this.bubble) this.flush();
  },
  flush() {
    this.shown = this.target.length;
    this._render();
    this.stop();
  },
  tick() {
    if (!this.bubble) return this.stop();
    const n = Math.max(1, Math.ceil((this.target.length - this.shown) / 120));
    this.shown = Math.min(this.target.length, this.shown + n);
    this._render();
    if (this.final && this.shown >= this.target.length) this.stop();
  },
  _render() {
    const text = this.target.slice(0, this.shown);
    this.bubble.dataset.raw = text;
    this.bubble.innerHTML = renderMD(text);
    linkFiles(this.bubble);
    this.bubble.classList.toggle("caret", text.length === 0);
    scrollBottom();
  },
  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    if (this.bubble) {
      this.bubble.classList.remove("streaming", "caret");
      this.bubble = null;
    }
  },
};

let abortCtl = null;

function setComposerLocked(locked) {
  state.streaming = locked;
  els.input.disabled = locked;
  els.send.textContent = locked ? "Stop" : "Send";
  els.send.disabled = !locked && !els.input.value.trim();
  els.send.classList.toggle("stop-mode", locked);
  els.gen.hidden = !locked;
  els.quick.classList.toggle("hidden", locked);
}

function mode() {
  const m = $('input[name="cb-mode"]:checked');
  return m ? m.value : "Quick";
}

function setBusy(busy) {
  els.index.disabled = busy;
  els.clear.disabled = busy;
  els.export.disabled = busy;
  if (busy) els.status.classList.add("loading");
  else els.status.classList.remove("loading");
}

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = els.input.scrollHeight + "px";
  els.send.disabled = state.streaming ? false : !els.input.value.trim();
}

function saveSession() {
  try {
    if (state.currentId) {
      state.sessions[state.currentId] = {
        name: state.sessions[state.currentId]?.name || els.source.value.trim() || "Untitled",
        history: state.history,
        summary: state.summary,
        source: els.source.value.trim(),
      };
    }
    localStorage.setItem(
      LS_KEY,
      JSON.stringify({ current: state.currentId, sessions: state.sessions })
    );
  } catch (e) {
    /* ignore quota/private-mode errors */
  }
}

function loadSession() {
  let s = null;
  try {
    s = JSON.parse(localStorage.getItem(LS_KEY) || "null");
  } catch (e) {
    return;
  }
  if (!s || !s.sessions) {
    migrateV3(s);
    return;
  }
  state.sessions = s.sessions;
  state.currentId = s.current || Object.keys(state.sessions)[0] || "";
  hydrateSessions();
  hydrateCurrent();
}

function migrateV3(s) {
  const data = s && typeof s === "object" ? s : null;
  if (!data || !(Array.isArray(data.history) && data.history.length)) {
    newSession();
    return;
  }
  const id = newId();
  state.sessions[id] = { name: data.source || "Migrated", history: data.history, summary: data.summary || "", source: data.source || "" };
  state.currentId = id;
  hydrateSessions();
  hydrateCurrent();
  saveSession();
  try { localStorage.removeItem("codebase-qa.session.v3"); } catch (e) { /* ignore */ }
}

function newSession() {
  const id = newId();
  state.currentId = id;
  state.sessions[id] = { name: els.source.value.trim() || "Untitled", history: [], summary: "", source: els.source.value.trim() };
  clearChat();
  saveSession();
  hydrateSessions();
  els.input.focus();
}

function switchSession(id) {
  if (id === state.currentId || !state.sessions[id]) return;
  saveSession();
  state.currentId = id;
  hydrateCurrent();
  hydrateSessions();
  saveSession();
}

function deleteSession(id) {
  if (Object.keys(state.sessions).length <= 1) return;
  delete state.sessions[id];
  if (state.currentId === id) state.currentId = Object.keys(state.sessions)[0] || "";
  hydrateCurrent();
  hydrateSessions();
  saveSession();
}

function renameSession(id, newName) {
  if (!state.sessions[id]) return;
  state.sessions[id].name = newName || state.sessions[id].name;
  saveSession();
  hydrateSessions();
}

function hydrateCurrent() {
  const sess = state.sessions[state.currentId] || { history: [], summary: "", source: "" };
  state.history = sess.history || [];
  state.summary = sess.summary || "";
  els.source.value = sess.source || "";
  els.messages.querySelectorAll(".msg").forEach((n) => n.remove());
  state.history.forEach((m) => addMessage(m.role, m.content || ""));
  els.messages.appendChild(els.empty);
  activeMsg = null;
  refreshEmpty();
  refreshRegen();
  scrollBottom(true);
}

function hydrateSessions() {
  const list = $("#cb-sessions");
  if (!list) return;
  list.innerHTML = "";
  Object.entries(state.sessions).forEach(([id, sess]) => {
    const row = document.createElement("div");
    row.className = "session" + (id === state.currentId ? " current" : "");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sname";
    btn.textContent = sess.name || "Untitled";
    btn.addEventListener("dblclick", () => startRename(row, id));
    btn.addEventListener("click", () => switchSession(id));
    const del = document.createElement("button");
    del.type = "button";
    del.className = "sdel";
    del.title = "Delete session";
    del.textContent = "\u00d7";
    del.addEventListener("click", (e) => { e.stopPropagation(); deleteSession(id); });
    row.append(btn, del);
    list.appendChild(row);
  });
}

function startRename(row, id) {
  if (row.querySelector("input")) return;
  const btn = row.querySelector(".sname");
  const input = document.createElement("input");
  input.type = "text";
  input.className = "srename";
  input.value = state.sessions[id]?.name || "";
  const finish = () => {
    renameSession(id, input.value.trim());
    input.replaceWith(btn);
  };
  input.addEventListener("blur", finish);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") input.blur();
    else if (e.key === "Escape") { input.value = state.sessions[id]?.name || ""; input.blur(); }
  });
  btn.replaceWith(input);
  input.focus();
  input.select();
}

function clearChat() {
  if (abortCtl) abortCtl.abort();
  typer.stop();
  abortCtl = null;
  state.history = [];
  state.summary = "";
  els.messages.querySelectorAll(".msg").forEach((n) => n.remove());
  els.messages.appendChild(els.empty);
  activeMsg = null;
  refreshEmpty();
  refreshRegen();
  saveSession();
}

function stopGen() {
  if (abortCtl) abortCtl.abort();
}

async function send() {
  const text = els.input.value.trim();
  if (!text || state.streaming) return;

  state.history.push({ role: "user", content: text });
  addMessage("user", text);
  const bot = addMessage("bot", "");
  typer.finishNow();
  typer.start(bot);
  els.input.value = "";
  autosize();
  setComposerLocked(true);
  stick = true;

  const ctl = new AbortController();
  abortCtl = ctl;
  try {
    let res;
    try {
      res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: state.history.slice(0, -1),
          mode: mode(),
          summary: state.summary,
        }),
        signal: ctl.signal,
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
    } catch (e) {
      throw e;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        if (!raw.startsWith("data:")) continue;
        let ev;
        try {
          ev = JSON.parse(raw.slice(5));
        } catch {
          continue;
        }
        if (ev.type === "delta") {
          typer.append(ev.text);
        } else if (ev.type === "step") {
          if (ev.text) addStep(ev.text);
        } else if (ev.type === "sources") {
          renderSources(ev.items);
        } else if (ev.type === "error") {
          typer.force(ev.text);
        } else if (ev.type === "done") {
          typer.finalize();
          state.history = ev.history || state.history;
          state.summary = ev.summary || "";
          if (ev.sources) renderSources(ev.sources);
        }
      }
    }
  } catch (e) {
    if (ctl.signal.aborted) {
      const last = state.history[state.history.length - 1];
      if (last && last.role === "assistant") last.content = typer.target || last.content;
      else state.history.push({ role: "assistant", content: typer.target || "" });
      typer.finalize();
    } else if (typer.bubble) {
      typer.force(typer.target || "Error: connection lost.");
    }
  } finally {
    abortCtl = null;
    autosize();
    setComposerLocked(false);
    refreshEmpty();
    refreshRegen();
    saveSession();
  }
}

function regenerate() {
  if (state.streaming || !state.history.length) return;
  const hist = state.history;
  const li = hist.map((h) => h.role).lastIndexOf("assistant");
  if (li <= 0) return;
  const text = hist[li - 1].content || "";
  if (!text) return;
  const rows = $$(".msg");
  if (rows[li]) rows[li].remove();
  if (rows[li - 1]) rows[li - 1].remove();
  hist.splice(li - 1, 2);
  refreshEmpty();
  refreshRegen();
  els.input.value = text;
  autosize();
  send();
}

let pollId = null;
function startPoll() {
  if (pollId) return;
  pollId = setInterval(async () => {
    try {
      const r = await fetch("/api/index/status");
      const d = await r.json();
      if (d && d.phase) els.status.textContent = d.phase;
    } catch (e) {
      /* skip */
    }
  }, 400);
}
function stopPoll() {
  if (pollId) {
    clearInterval(pollId);
    pollId = null;
  }
}

async function indexRepo() {
  const source = els.source.value.trim();
  if (!source || els.index.disabled) return;
  setBusy(true);
  els.status.textContent = "Indexing…";
  startPoll();
  try {
    const res = await fetch("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    const data = await res.json();
    stopPoll();
    els.status.textContent = data.status || res.statusText;
    state.files = Array.isArray(data.files) ? data.files : state.files;
    $$(".msg .bubble").forEach((b) => linkFiles(b));
    if (data.graph_html) {
      state.graphHtml = data.graph_html;
      mountGraph();
    }
    saveSession();
  } catch (e) {
    stopPoll();
    els.status.textContent = "Error: " + e.message;
  } finally {
    stopPoll();
    setBusy(false);
  }
}

async function clearIndex() {
  if (els.clear.disabled) return;
  setBusy(true);
  try {
    const res = await fetch("/api/clear", { method: "POST" });
    const data = await res.json();
    els.status.textContent = data.status || "Index cleared.";
    state.graphHtml = "";
    state.files = [];
  } catch (e) {
    els.status.textContent = "Error: " + e.message;
  } finally {
    setBusy(false);
  }
}

function mountGraph() {
  if (!state.graphHtml) return;
  els.graphBody.style.display = "grid";
  els.graphBody.innerHTML = "";
  let html = state.graphHtml;
  const wrapped = new DOMParser().parseFromString(html, "text/html").querySelector("iframe");
  if (wrapped && wrapped.getAttribute("srcdoc")) html = wrapped.getAttribute("srcdoc");
  const iframe = document.createElement("iframe");
  iframe.srcdoc = html;
  iframe.setAttribute("title", "Dependency graph");
  els.graphBody.appendChild(iframe);
}

async function showGraph() {
  if (!state.graphHtml) {
    els.graphBody.innerHTML =
      '<p class="hint">Index a repository to view its dependency graph.</p>';
    try {
      const res = await fetch("/api/graph");
      const data = await res.json();
      state.graphHtml = data.graph_html || "";
      mountGraph();
    } catch {
      /* keep hint */
    }
  } else {
    mountGraph();
  }
}

async function exportChat() {
  if (els.export.disabled || !state.history.length) return;
  els.export.disabled = true;
  try {
    const res = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format_type: els.format.value, history: state.history }),
    });
    const data = await res.json();
    if (!data.content) return;
    const blob = new Blob([data.content], {
      type: "text/plain;charset=utf-8",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = data.filename || "session." + (els.format.value === "Notebook" ? "ipynb" : "md");
    a.click();
    URL.revokeObjectURL(a.href);
  } finally {
    els.export.disabled = false;
  }
}

async function fetchInfo() {
  try {
    const r = await fetch("/api/info");
    const d = await r.json();
    if (!d.provider) return;
    const line = `Provider: ${d.provider} · Model: ${d.model} · Embedding: ${d.embedding} · Retrieval K: ${d.retrieval_k}`;
    if (!els.status.textContent.trim()) els.status.textContent = line;
    if (els.footerInfo) els.footerInfo.textContent = `${d.provider} · ${d.model}`;
  } catch (e) {
    /* skip */
  }
}

window.addEventListener("message", (e) => {
  const msg = e.data || {};
  if (msg.type === "cb-open-file" && msg.path) openFile(msg.path);
  else if (msg.type === "cb-ask" && msg.text) {
    switchTo("chat");
    els.input.value = msg.text;
    autosize();
    send();
  }
});

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  if (state.streaming) stopGen();
  else send();
});

els.input.addEventListener("input", autosize);
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

els.quick.addEventListener("click", (e) => {
  const b = e.target.closest(".pill");
  if (!b) return;
  els.input.value = b.textContent;
  autosize();
  els.input.focus();
});

els.messages.addEventListener("click", (e) => {
  const src = e.target.closest(".src");
  if (src && src.dataset.path) {
    openFile(src.dataset.path);
    return;
  }
  const fl = e.target.closest(".filelink");
  if (fl && fl.dataset.path) {
    e.preventDefault();
    openFile(fl.dataset.path);
    return;
  }
  const act = e.target.closest(".act");
  if (!act) return;
  const row = act.closest(".msg");
  const raw = (row && row.querySelector(".bubble")) ? row.querySelector(".bubble").dataset.raw || "" : "";
  if (act.dataset.act === "copy") copyText(raw, act);
  else if (act.dataset.act === "regen") regenerate();
});

els.messages.addEventListener("scroll", () => {
  if (!nearBottom()) stick = false;
});

els.index.addEventListener("click", indexRepo);
els.clear.addEventListener("click", clearIndex);
els.export.addEventListener("click", exportChat);
els.newchat.addEventListener("click", () => {
  newSession();
});
els.clearchat.addEventListener("click", clearChat);
els.newsession.addEventListener("click", newSession);
els.source.addEventListener("change", saveSession);
$$('input[name="cb-mode"]').forEach((r) => r.addEventListener("change", saveSession));

window.addEventListener("keydown", (e) => {
  const mod = e.ctrlKey || e.metaKey;
  const k = e.key.toLowerCase();
  if (mod && k === "n") {
    e.preventDefault();
    newSession();
  } else if (mod && k === "k") {
    e.preventDefault();
    switchTo("search");
  } else if (e.key === "Escape") {
    if (!els.overlay.hidden) closeOverlay();
    else els.searchInput.blur();
  }
});

els.viewtabs.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) switchTo(b.dataset.view);
});

els.searchInput.addEventListener("input", runSearch);
els.searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    runSearch(true);
  }
});

els.searchResults.addEventListener("click", (e) => {
  const btn = e.target.closest(".act");
  if (!btn || !btn.dataset.path) return;
  if (btn.dataset.a === "open") openFile(btn.dataset.path);
  else if (btn.dataset.a === "ask") askAbout(btn.dataset.path);
});

els.overlayClose.addEventListener("click", closeOverlay);
els.overlayAsk.addEventListener("click", () => askAbout(els.overlayAsk.dataset.path || ""));
els.overlay.addEventListener("click", (e) => {
  if (e.target === els.overlay) closeOverlay();
});

els.viewChat.addEventListener("click", (e) => {
  if (e.target === els.input || e.target === els.send) return;
  if (e.target.closest("a, button, input, select, textarea, iframe")) return;
  els.input.focus();
});

$$(".panel[data-acc]").forEach((panel) => {
  const head = panel.querySelector(".acchead");
  const sig = panel.querySelector(".accsig");
  const toggle = () => {
    const closed = panel.classList.toggle("closed");
    sig.textContent = closed ? "+" : "\u2212";
  };
  head.addEventListener("click", toggle);
  head.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggle();
    }
  });
});

if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((ent) => {
        if (ent.isIntersecting) {
          ent.target.classList.add("in");
          io.unobserve(ent.target);
        }
      });
    },
    { threshold: 0.06 }
  );
  $$(".reveal").forEach((el) => io.observe(el));
} else {
  $$(".reveal").forEach((el) => el.classList.add("in"));
}

loadSession();
autosize();
refreshEmpty();
refreshRegen();
fetchInfo();