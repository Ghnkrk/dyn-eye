/**
 * DYN-EYE Dashboard — Frontend v2
 *
 * Features:
 *  - SSE-based real-time log streaming (no polling!)
 *  - Light/Dark theme toggle with persistence
 *  - Pipeline status tracking from live logs
 *  - Fully autonomous — minimal user interaction
 */

const API = '';

// ── Theme ───────────────────────────────────────────────────
function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    document.getElementById('theme-btn').textContent = next === 'dark' ? '☀️' : '🌙';
}

function loadTheme() {
    const saved = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);
    document.getElementById('theme-btn').textContent = saved === 'dark' ? '☀️' : '🌙';
}

// ── Toast ────────────────────────────────────────────────────
function toast(msg, type = 'info') {
    const w = document.getElementById('toast-wrap');
    const t = document.createElement('div');
    t.className = `toast toast--${type === 'success' ? 'ok' : type === 'error' ? 'err' : 'info'}`;
    t.textContent = msg;
    w.appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 250); }, 3500);
}

// ── API Helpers ──────────────────────────────────────────────
async function get(path) {
    const r = await fetch(`${API}${path}`);
    if (!r.ok) throw new Error(r.statusText);
    return r.json();
}

async function post(path, body = {}) {
    const r = await fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!r.ok) {
        const e = await r.json().catch(() => ({}));
        throw new Error(e.detail || r.statusText);
    }
    return r.json();
}

// ── SSE Log Stream ──────────────────────────────────────────
let eventSource = null;
let logAutoScroll = true;

function startLogStream() {
    if (eventSource) eventSource.close();

    eventSource = new EventSource('/api/logs/stream');
    eventSource.onmessage = (e) => {
        const evt = JSON.parse(e.data);
        appendLog(evt);
        updatePipelineFromLog(evt);
    };
    eventSource.onerror = () => {
        document.getElementById('log-dot').style.background = 'var(--error)';
        // Auto-reconnect after 3s
        setTimeout(() => {
            if (eventSource.readyState === EventSource.CLOSED) startLogStream();
        }, 3000);
    };
    eventSource.onopen = () => {
        document.getElementById('log-dot').style.background = 'var(--success)';
    };
}

function appendLog(evt) {
    const body = document.getElementById('log-body');
    const line = document.createElement('div');
    line.className = `log-line log-line--${evt.level}`;

    const time = new Date(evt.ts).toLocaleTimeString('en-US', { hour12: false });
    line.innerHTML = `
        <span class="log-ts">${time}</span>
        <span class="log-src">${evt.source}</span>
        <span class="log-msg">${escapeHtml(evt.message)}</span>
    `;
    body.appendChild(line);

    // Auto-scroll if user hasn't scrolled up
    if (logAutoScroll) {
        body.scrollTop = body.scrollHeight;
    }

    // Keep max 500 lines in DOM
    while (body.children.length > 500) body.removeChild(body.firstChild);
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// Track if user scrolled up
document.addEventListener('DOMContentLoaded', () => {
    const body = document.getElementById('log-body');
    body.addEventListener('scroll', () => {
        logAutoScroll = body.scrollTop + body.clientHeight >= body.scrollHeight - 30;
    });
});

function clearLogs() {
    const body = document.getElementById('log-body');
    body.innerHTML = '';
}

// ── Pipeline Status from Logs ───────────────────────────────
const NODE_NAMES = [
    'yolo_inference', 'vlm_annotation', 'crop_extraction',
    'feature_extraction', 'faiss_search', 'hdbscan_cluster',
    'label_studio_sync',
];

function updatePipelineFromLog(evt) {
    const source = evt.source;

    // Update global status
    if (evt.message.includes('pipeline triggered') || evt.message.includes('pipeline started')) {
        setGlobalStatus('running', 'Running');
        setDiscoveryStatus('running', 'Running');
        // Reset all nodes
        NODE_NAMES.forEach(n => setNodeState(n, ''));
    }

    if (evt.message.includes('pipeline finished') || evt.message.includes('pipeline complete')) {
        setGlobalStatus('complete', 'Complete');
        setDiscoveryStatus('complete', 'Complete');
        document.getElementById('btn-discover').disabled = false;
        loadStats();
    }

    if (evt.message.includes('pipeline failed')) {
        setGlobalStatus('failed', 'Failed');
        setDiscoveryStatus('failed', 'Failed');
        document.getElementById('btn-discover').disabled = false;
    }

    // Update individual nodes
    if (NODE_NAMES.includes(source)) {
        if (evt.level === 'step' && evt.message.includes('Starting')) {
            setNodeState(source, 'active');
        }
        if (evt.level === 'info' && evt.message.includes('complete')) {
            setNodeState(source, 'done');
            // Extract items from data
            if (evt.data && evt.data.items_processed !== undefined) {
                const detail = document.getElementById(`pd-${source}`);
                if (detail) detail.textContent = `${evt.data.items_processed} items`;
            }
        }
        if (evt.level === 'error') {
            setNodeState(source, 'fail');
        }
    }
}

function setNodeState(name, state) {
    const el = document.getElementById(`pn-${name}`);
    if (!el) return;
    el.classList.remove('pipe-node--active', 'pipe-node--done', 'pipe-node--fail');
    if (state) el.classList.add(`pipe-node--${state}`);
}

function setGlobalStatus(status, text) {
    const pill = document.getElementById('global-pill');
    pill.className = `status-pill status-pill--${status}`;
    document.getElementById('global-text').textContent = text;
}

function setDiscoveryStatus(status, text) {
    const pill = document.getElementById('discovery-pill');
    pill.className = `status-pill status-pill--${status}`;
    document.getElementById('disc-status-text').textContent = text;
}

// ── Triggers ────────────────────────────────────────────────
async function triggerDiscovery() {
    const btn = document.getElementById('btn-discover');
    btn.disabled = true;

    const conf = parseFloat(document.getElementById('in-conf').value) || null;
    const useSampleRun = document.getElementById('in-sample-run').checked;

    try {
        await post('/api/discovery/trigger', {
            confidence_threshold: conf,
            use_sample_run: useSampleRun,
        });
        toast('Discovery pipeline started', 'success');
    } catch (e) {
        toast(`Failed: ${e.message}`, 'error');
        btn.disabled = false;
    }
}

async function setupFAISS() {
    try {
        toast('Building FAISS index...', 'info');
        const r = await post('/api/faiss/setup', {});
        toast(`FAISS: ${r.count} vectors indexed`, 'success');
    } catch (e) {
        toast(`FAISS failed: ${e.message}`, 'error');
    }
}

let orchRunning = false;

async function toggleOrchestrator() {
    const btn = document.getElementById('btn-orch');
    try {
        if (!orchRunning) {
            await post('/api/orchestrator/start', { project_id: 1 });
            orchRunning = true;
            btn.textContent = '🛑 Stop Orchestrator';
            btn.classList.remove('btn--outline');
            btn.classList.add('btn--danger');
            toast('Orchestrator started', 'success');
        } else {
            await post('/api/orchestrator/stop');
            orchRunning = false;
            btn.textContent = '🤖 Start Orchestrator';
            btn.classList.remove('btn--danger');
            btn.classList.add('btn--outline');
            toast('Orchestrator stopped', 'info');
        }
    } catch (e) {
        toast(`Orchestrator: ${e.message}`, 'error');
    }
}

// ── Data Loading ────────────────────────────────────────────
async function loadStats() {
    try {
        const [cfg, clusters, versions, runs] = await Promise.all([
            get('/api/config'),
            get('/api/clusters'),
            get('/api/model-versions'),
            get('/api/runs'),
        ]);

        document.getElementById('s-images').textContent = cfg.input_images_count || '0';
        document.getElementById('s-clusters').textContent = (clusters.clusters || []).length || '0';
        document.getElementById('s-models').textContent = (versions.versions || []).length || '0';
        document.getElementById('s-runs').textContent = (runs || []).length || '0';

        const known = cfg.known_defect_names || [];
        document.getElementById('cfg-known-defects').textContent = known.length ? known.join(', ') : 'None';

        // Render clusters
        renderClusters(clusters.clusters || []);
    } catch (e) {
        console.error('Stats load failed:', e);
    }
}

function renderClusters(clusters) {
    window.lastLoadedClusters = clusters;
    const grid = document.getElementById('clusters-grid');
    if (!clusters.length) {
        grid.innerHTML = `
            <div class="empty" style="grid-column:1/-1">
                <div class="empty-icon">🔬</div>
                <div>Run the pipeline to discover clusters</div>
            </div>`;
        return;
    }

    let html = '';
    for (const c of clusters) {
        const isNoise = c.name === 'noise';
        const thumbs = (c.images || []).slice(0, 4).map(img =>
            `<img src="/api/clusters/${c.name}/${img}" class="cluster-thumb" alt="${img}" />`
        ).join('');

        html += `
        <div class="cluster-card${isNoise ? ' cluster-card--noise' : ''}">
            <div class="cluster-card-header" onclick="openClusterModal('${c.name}')" style="cursor:pointer;" title="Click to view and edit cluster images">
                <span class="cluster-badge">${c.name}</span>
                <span class="cluster-count-badge">${c.image_count} crops</span>
            </div>
            <div class="cluster-thumbs" onclick="openClusterModal('${c.name}')" style="cursor:pointer;" title="Click to view and edit cluster images">${thumbs}</div>
            ${isNoise ? '' : `<div class="cluster-name-row">
                <input type="text" class="cluster-name-input" id="cname-${c.name}"
                       placeholder="Enter defect name…" spellcheck="false" />
            </div>`}
        </div>`;
    }

    html += `
    <div style="grid-column:1/-1; display:flex; gap:8px; align-items:center; margin-top:4px; flex-wrap:wrap;">
        <button class="btn btn--primary" onclick="submitClusterNames()">
            💾 Save Cluster Names
        </button>
        <button class="btn btn--outline btn--sm" onclick="pullFromLabelStudio()">
            🔄 Pull from Label Studio
        </button>
        <span class="cluster-save-status" id="cluster-save-status"></span>
    </div>`;

    grid.innerHTML = html;
}

async function submitClusterNames() {
    const inputs = document.querySelectorAll('.cluster-name-input');
    const names = {};
    let empty = 0;

    inputs.forEach(inp => {
        const clusterId = inp.id.replace('cname-', '');
        const val = inp.value.trim();
        if (val) {
            names[clusterId] = val;
        } else {
            empty++;
        }
    });

    if (Object.keys(names).length === 0) {
        toast('Enter at least one cluster name', 'error');
        return;
    }

    try {
        const r = await post('/api/clusters/name', { names });
        toast(`Saved ${r.updated_clusters.length} cluster names → ${r.total_labeled_crops} crops labeled`, 'success');
        document.getElementById('cluster-save-status').textContent =
            `✓ ${r.total_labeled_crops} crops mapped to defect labels`;
        document.getElementById('cluster-save-status').style.color = 'var(--success)';
    } catch (e) {
        toast(`Failed to save names: ${e.message}`, 'error');
    }
}

async function pullFromLabelStudio() {
    toast('Pulling annotations from Label Studio...', 'info');
    try {
        const r = await post('/api/clusters/pull-from-ls');
        const parts = [];
        if (Object.keys(r.defect_names || {}).length) {
            parts.push(`${Object.keys(r.defect_names).length} named`);
        }
        if (r.reassignments) parts.push(`${r.reassignments} reassigned`);
        if (r.drops) parts.push(`${r.drops} dropped`);
        parts.push(`${r.total_labeled_crops} total labeled`);

        toast(`LS sync: ${parts.join(', ')}`, 'success');
        document.getElementById('cluster-save-status').textContent =
            `✓ Synced from LS — ${parts.join(', ')}`;
        document.getElementById('cluster-save-status').style.color = 'var(--success)';
        loadStats(); // Refresh cluster cards
    } catch (e) {
        toast(`LS pull failed: ${e.message}`, 'error');
    }
}

// ── Status Polling (lightweight, supplements SSE) ───────────
async function pollStatus() {
    try {
        const s = await get('/api/status');
        // Update orchestrator button state
        if (s.orchestrator && s.orchestrator.status === 'running') {
            orchRunning = true;
            const btn = document.getElementById('btn-orch');
            btn.textContent = '🛑 Stop Orchestrator';
            btn.classList.remove('btn--outline');
            btn.classList.add('btn--danger');
        }

        // Update discovery button
        if (s.discovery) {
            if (s.discovery.status === 'running') {
                document.getElementById('btn-discover').disabled = true;
                setDiscoveryStatus('running', 'Running');
                setGlobalStatus('running', 'Running');
            } else if (s.discovery.status !== 'idle') {
                document.getElementById('btn-discover').disabled = false;
            }
        }
    } catch (e) { /* silent */ }
}

// ── Load recent logs on page load ───────────────────────────
async function loadRecentLogs() {
    try {
        const logs = await get('/api/logs/recent?n=50');
        logs.forEach(evt => appendLog(evt));
    } catch (e) { /* silent */ }
}

// ── Interactive Cluster Crop Editor Modal ───────────────────
function openClusterModal(clusterName) {
    const clusters = window.lastLoadedClusters || [];
    const cluster = clusters.find(c => c.name === clusterName);
    if (!cluster) {
        toast('Cluster data not found', 'error');
        return;
    }

    const modal = document.getElementById('cluster-modal');
    const title = document.getElementById('cluster-modal-title');
    const body = document.getElementById('cluster-modal-body');

    title.textContent = `Manage Cluster: ${clusterName} (${cluster.image_count} crops)`;
    
    // Build options for target clusters re-assignment
    const otherClusters = clusters.map(c => c.name);
    
    // Build batch select options
    const batchOptions = otherClusters.map(cname => {
        const selected = cname === clusterName ? ' selected' : '';
        return `<option value="${cname}"${selected}>${cname}</option>`;
    }).join('');

    let html = `
    <!-- Batch Action Bar -->
    <div class="batch-bar" id="batch-bar" style="display:none; justify-content: space-between; align-items: center; padding: 12px; background: rgba(30, 41, 59, 0.9); backdrop-filter: blur(10px); border-radius: 8px; margin-bottom: 16px; border: 1px solid rgba(255,255,255,0.15);">
      <div style="color:var(--text-primary); font-weight: 500; font-size:14px;"><span id="batch-count">0</span> crops selected</div>
      <div style="display:flex; gap:8px; align-items:center;">
        <span style="font-size:13px; color:var(--text-secondary);">Move to:</span>
        <select id="batch-target-select" style="padding:6px 12px; border-radius:4px; background:var(--bg-secondary); color:var(--text-primary); border:1px solid rgba(255,255,255,0.15); cursor:pointer; font-size:13px;">
          ${batchOptions}
        </select>
        <button class="btn btn--primary btn--sm" onclick="runBatchMove('${clusterName}')" style="padding: 6px 12px; font-size: 13px;">Apply Move</button>
        <button class="btn btn--danger btn--sm" onclick="runBatchDrop('${clusterName}')" style="padding: 6px 12px; font-size: 13px; background: var(--error);">🗑️ Drop Selected</button>
      </div>
    </div>
    
    <div class="crop-grid">`;
    
    if (!cluster.images || cluster.images.length === 0) {
        html += `<div style="grid-column:1/-1; text-align:center; padding:20px; color:var(--text-secondary);">This cluster is empty.</div>`;
    } else {
        cluster.images.forEach(img => {
            // Dropdown options
            const options = otherClusters.map(cname => {
                const selected = cname === clusterName ? ' selected' : '';
                return `<option value="${cname}"${selected}>${cname}</option>`;
            }).join('');

            html += `
            <div class="crop-card" id="crop-card-${clusterName}-${img.replace(/\./g, '_')}" style="position: relative;">
                <input type="checkbox" class="crop-checkbox" data-crop="${img}" onchange="updateBatchUI('${clusterName}')" style="position: absolute; top: 8px; left: 8px; z-index: 10; width: 18px; height: 18px; cursor: pointer; accent-color: var(--success);" />
                <div class="crop-card-img-wrap">
                    <img src="/api/clusters/${clusterName}/${img}" class="crop-card-img" alt="${img}" />
                </div>
                <div class="crop-card-actions">
                    <select class="crop-select" onchange="moveCrop('${clusterName}', '${img}', this.value)">
                        ${options}
                    </select>
                    <button class="crop-btn-drop" onclick="dropCrop('${clusterName}', '${img}')">
                        🗑️ Drop
                    </button>
                </div>
            </div>`;
        });
    }
    
    html += '</div>';
    body.innerHTML = html;

    modal.classList.add('active');
}

function updateBatchUI(clusterName) {
    const checked = document.querySelectorAll('.crop-checkbox:checked');
    const bar = document.getElementById('batch-bar');
    const count = document.getElementById('batch-count');
    if (checked.length > 0) {
        bar.style.display = 'flex';
        count.textContent = checked.length;
    } else {
        bar.style.display = 'none';
    }
}

async function runBatchMove(sourceCluster) {
    const checked = document.querySelectorAll('.crop-checkbox:checked');
    const cropFiles = Array.from(checked).map(cb => cb.dataset.crop);
    const targetCluster = document.getElementById('batch-target-select').value;
    
    if (sourceCluster === targetCluster) {
        toast('Cannot move crops to the same cluster', 'error');
        return;
    }
    
    try {
        toast(`Moving ${cropFiles.length} crops to ${targetCluster}...`, 'info');
        await post('/api/clusters/batch-edit-crops', {
            crop_files: cropFiles,
            source_cluster: sourceCluster,
            target_cluster: targetCluster,
            action: 'move'
        });
        toast(`Successfully moved ${cropFiles.length} crops`, 'success');
        openClusterModal(sourceCluster);
    } catch (e) {
        toast(`Batch move failed: ${e.message}`, 'error');
    }
}

async function runBatchDrop(sourceCluster) {
    const checked = document.querySelectorAll('.crop-checkbox:checked');
    const cropFiles = Array.from(checked).map(cb => cb.dataset.crop);
    
    if (!confirm(`Are you sure you want to drop these ${cropFiles.length} crops?`)) {
        return;
    }
    
    try {
        toast(`Dropping ${cropFiles.length} crops...`, 'info');
        await post('/api/clusters/batch-edit-crops', {
            crop_files: cropFiles,
            source_cluster: sourceCluster,
            target_cluster: null,
            action: 'drop'
        });
        toast(`Successfully dropped ${cropFiles.length} crops`, 'success');
        openClusterModal(sourceCluster);
    } catch (e) {
        toast(`Batch drop failed: ${e.message}`, 'error');
    }
}

function closeClusterModal() {
    const modal = document.getElementById('cluster-modal');
    modal.classList.remove('active');
    loadStats(); // Reload dashboard clusters in the background
}

async function moveCrop(sourceCluster, cropFile, targetCluster) {
    if (sourceCluster === targetCluster) return;
    try {
        toast(`Moving crop to ${targetCluster}...`, 'info');
        const res = await post('/api/clusters/edit-crop', {
            crop_file: cropFile,
            source_cluster: sourceCluster,
            target_cluster: targetCluster,
            action: 'move'
        });
        toast('Moved crop successfully', 'success');
        // Instantly reload modal with source cluster to show it's gone
        openClusterModal(sourceCluster);
    } catch (e) {
        toast(`Move failed: ${e.message}`, 'error');
    }
}

async function dropCrop(sourceCluster, cropFile) {
    if (!confirm('Are you sure you want to drop this crop? It will be removed from retraining dataset.')) {
        return;
    }
    try {
        toast('Dropping crop...', 'info');
        const res = await post('/api/clusters/edit-crop', {
            crop_file: cropFile,
            source_cluster: sourceCluster,
            target_cluster: null,
            action: 'drop'
        });
        toast('Dropped crop successfully', 'success');
        // Instantly reload modal
        openClusterModal(sourceCluster);
    } catch (e) {
        toast(`Drop failed: ${e.message}`, 'error');
    }
}

async function triggerRetraining() {
    const btn = document.getElementById('btn-retrain');
    btn.disabled = true;
    try {
        toast('Triggering Retraining Agent...', 'info');
        const r = await post('/api/retraining/trigger', {
            project_id: -1,
            epochs: 2,
            imgsz: 640,
            batch_size: 2
        });
        toast('Retraining pipeline triggered successfully!', 'success');
    } catch (e) {
        toast(`Retraining failed: ${e.message}`, 'error');
    } finally {
        setTimeout(() => { btn.disabled = false; }, 5000);
    }
}


// ── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
    loadStats();
    loadRecentLogs();
    startLogStream();
    pollStatus();

    // Periodic stats refresh (every 15s)
    setInterval(loadStats, 15000);
    setInterval(pollStatus, 5000);
});
