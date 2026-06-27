'use strict';

const csrfToken = () => document.getElementById('csrf-token').value;

function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function showStatus(el, msg, hidden = false) {
  el.textContent = msg;
  el.hidden = hidden || !msg;
}

// ── Query form ────────────────────────────────────────────────────────────────

const queryForm   = document.getElementById('query-form');
const queryInput  = document.getElementById('query-input');
const querySubmit = document.getElementById('query-submit');
const queryStatus = document.getElementById('query-status');
const resultBox   = document.getElementById('result-box');
const resultMeta  = document.getElementById('result-meta');
const resultText  = document.getElementById('result-text');

queryForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;

  querySubmit.disabled = true;
  showStatus(queryStatus, 'Researching — this may take up to 90 seconds…');
  resultBox.hidden = true;

  try {
    const resp = await fetch('/ui/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken(),
      },
      body: JSON.stringify({ query }),
    });
    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`${resp.status}: ${body}`);
    }
    const data = await resp.json();
    showStatus(queryStatus, '', true);
    resultMeta.textContent = `Confidence: ${data.confidence}/5`;
    resultText.textContent = data.answer;
    resultBox.hidden = false;
    loadHistory();
  } catch (err) {
    showStatus(queryStatus, `Error: ${err.message}`);
  } finally {
    querySubmit.disabled = false;
  }
});

// ── File upload ───────────────────────────────────────────────────────────────

const uploadForm   = document.getElementById('upload-form');
const fileInput    = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');

uploadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  showStatus(uploadStatus, 'Uploading…');

  try {
    const resp = await fetch('/ui/ingest', {
      method: 'POST',
      headers: { 'X-CSRF-Token': csrfToken() },
      body: formData,
    });
    if (!resp.ok) {
      const body = await resp.text();
      throw new Error(`${resp.status}: ${body}`);
    }
    const data = await resp.json();
    fileInput.value = '';
    pollJob(data.job_id);
  } catch (err) {
    showStatus(uploadStatus, `Upload failed: ${err.message}`);
  }
});

function pollJob(jobId) {
  showStatus(uploadStatus, 'Indexing…');
  const iv = setInterval(async () => {
    try {
      const resp = await fetch(`/ui/ingest/status/${jobId}`);
      if (!resp.ok) {
        clearInterval(iv);
        showStatus(uploadStatus, `Status check failed (${resp.status})`);
        return;
      }
      const job = await resp.json();
      if (job.status === 'done') {
        clearInterval(iv);
        showStatus(uploadStatus, `Indexed: ${job.filename}`);
      } else if (job.status === 'failed') {
        clearInterval(iv);
        showStatus(uploadStatus, `Indexing failed: ${job.error || 'unknown error'}`);
      }
    } catch {
      clearInterval(iv);
      showStatus(uploadStatus, 'Could not retrieve job status.');
    }
  }, 2000);
}

// ── History ───────────────────────────────────────────────────────────────────

const historyBody  = document.getElementById('history-body');
const historyEmpty = document.getElementById('history-empty');
const historyTable = document.getElementById('history-table');

historyBody.addEventListener('click', async (e) => {
  const btn = e.target.closest('.del-btn');
  if (!btn) return;
  const id = Number(btn.dataset.id);
  if (!id) return;
  btn.disabled = true;
  try {
    await fetch(`/ui/history/${id}`, {
      method: 'DELETE',
      headers: { 'X-CSRF-Token': csrfToken() },
    });
    loadHistory();
  } catch {
    btn.disabled = false;
  }
});

async function loadHistory() {
  try {
    const resp = await fetch('/ui/history');
    if (!resp.ok) {
      if (resp.status === 401) window.location.href = '/login';
      return;
    }
    const items = await resp.json();
    historyBody.innerHTML = items.map((item) => `
      <tr>
        <td>${escHtml(item.created_at.slice(0, 16).replace('T', ' '))}</td>
        <td title="${escHtml(item.answer)}">${escHtml(item.query.slice(0, 80))}${item.query.length > 80 ? '…' : ''}</td>
        <td>${item.confidence ?? '—'}</td>
        <td><button class="del-btn" data-id="${item.id}">Delete</button></td>
      </tr>
    `).join('');
    const hasItems = items.length > 0;
    historyTable.hidden = !hasItems;
    historyEmpty.hidden = hasItems;
  } catch {
    // silently ignore — history is non-critical
  }
}

// ── Logout ────────────────────────────────────────────────────────────────────

document.getElementById('logout-btn').addEventListener('click', async () => {
  try {
    await fetch('/logout', {
      method: 'POST',
      headers: { 'X-CSRF-Token': csrfToken() },
    });
  } finally {
    window.location.href = '/login';
  }
});

// ── Init ──────────────────────────────────────────────────────────────────────

historyTable.hidden = true;
loadHistory();
