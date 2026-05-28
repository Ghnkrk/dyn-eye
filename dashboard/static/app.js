/**
 * DYN-EYE Dashboard — Frontend v3
 *
 * Features:
 *  - SSE-based real-time log streaming (no polling!)
 *  - Light/Dark theme toggle with persistence
 *  - Pipeline status tracking from live logs
 *  - Cache mode toggle for fast demo runs
 *  - In-dashboard cluster editing (no Label Studio needed)
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
let lastLogTs = null;
let hasDiscoveryRunThisSession = false;

function startLogStream() {
    if (eventSource) eventSource.close();

    const url = lastLogTs ? `/api/logs/stream?after_ts=${encodeURIComponent(lastLogTs)}` : '/api/logs/stream';
    eventSource = new EventSource(url);
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

let rightActiveTab = 'logs';

function switchRightTab(tabId) {
    rightActiveTab = tabId;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
    
    if (tabId === 'logs') {
        document.getElementById('tab-btn-logs').classList.add('active');
        document.getElementById('tab-content-logs').style.display = 'flex';
    } else {
        document.getElementById('tab-btn-train').classList.add('active');
        document.getElementById('tab-content-train').style.display = 'flex';
        // Hide red pulse dot
        document.getElementById('train-pulse-dot').style.display = 'none';
    }
}

function appendLog(evt) {
    if (!lastLogTs || evt.ts > lastLogTs) {
        lastLogTs = evt.ts;
    }

    // Intercept progress events (e.g. VLM annotation sequential image progress) for discovery nodes
    if (evt.level === 'progress' && NODE_NAMES.includes(evt.source)) {
        const nodeId = SOURCE_TO_NODE_ID[evt.source] || evt.source;
        const detail = document.getElementById(`pd-${nodeId}`);
        if (detail) {
            detail.textContent = evt.message;
        }
        return;
    }

    const isTrainLog = evt.source === 'yolo_train' || evt.source === 'train_yolo' || evt.source === 'model_registry' || evt.source === 'system' || evt.source === 'llm_advisor';
    const body = document.getElementById(isTrainLog ? 'train-log-body' : 'log-body');
    if (!body) return;

    // If first time receiving training logs, clear the default placeholder
    if (isTrainLog && body.textContent.includes('[YOLO Retraining Terminal Idle]')) {
        body.innerHTML = '';
        document.getElementById('train-dot').style.background = 'var(--success)';
    }

    const line = document.createElement('div');
    line.className = `log-line log-line--${evt.level}`;

    const time = new Date(evt.ts).toLocaleTimeString('en-US', { hour12: false });
    line.innerHTML = `
        <span class="log-ts">${time}</span>
        <span class="log-src">${evt.source}</span>
        <span class="log-msg">${escapeHtml(evt.message)}</span>
    `;
    body.appendChild(line);

    // Also mirror to the expanded terminal modal if it exists
    const modalBodyId = isTrainLog ? 'terminal-train-body' : 'terminal-logs-body';
    const modalBody = document.getElementById(modalBodyId);
    if (modalBody) {
        // Clear placeholder text first time
        if (isTrainLog && modalBody.textContent.includes('[Retraining Terminal Idle]')) {
            modalBody.innerHTML = '';
        }
        const lineClone = line.cloneNode(true);
        modalBody.appendChild(lineClone);
        modalBody.scrollTop = modalBody.scrollHeight;
        while (modalBody.children.length > 1000) modalBody.removeChild(modalBody.firstChild);
    }

    // Auto-scroll compact panel
    if (logAutoScroll) body.scrollTop = body.scrollHeight;

    // Parse training progress metrics if available
    if (isTrainLog) {
        // Toggle running wheel spinner based on active training logs
        const spinner = document.getElementById('training-spinner-widget');
        if (spinner) {
            if (evt.message.includes('Starting') || evt.message.includes('retraining pipeline started') || evt.message.includes('Epoch')) {
                spinner.style.display = 'flex';
            }
            if (evt.message.includes('finished') || evt.message.includes('complete') || evt.message.includes('deployed') || evt.message.includes('failed') || evt.message.includes('SYSTEM UNIVERSAL RESET')) {
                spinner.style.display = 'none';
            }
        }

        // Look for epoch messages: "Epoch 3/30" or similar
        const epochMatch = evt.message.match(/epoch\s+(\d+)\/(\d+)/i) || evt.message.match(/Epoch\s+(\d+)/);
        if (epochMatch) {
            document.getElementById('val-train-epoch').textContent = epochMatch[1] + (epochMatch[2] ? '/' + epochMatch[2] : '');
        }
        // Look for loss messages: "loss: 0.2312" or similar
        const lossMatch = evt.message.match(/loss:\s*([0-9.]+)/i);
        if (lossMatch) {
            document.getElementById('val-train-loss').textContent = parseFloat(lossMatch[1]).toFixed(4);
            document.getElementById('val-train-loss').style.color = 'var(--error)';
        }

        // Highlight tab with pulsing red dot if not currently focused
        if (rightActiveTab !== 'train') {
            document.getElementById('train-pulse-dot').style.display = 'inline-block';
        }

        // Check if training completed successfully
        if (evt.message.includes('retraining pipeline finished') || evt.message.includes('deployed successfully')) {
            document.getElementById('train-dot').style.background = 'var(--text-secondary)';
            loadModelVersions();
        }
    }

    // Keep max 500 lines in DOM
    while (body.children.length > 500) body.removeChild(body.firstChild);
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// Track scroll status
document.addEventListener('DOMContentLoaded', () => {
    const body = document.getElementById('log-body');
    if (body) {
        body.addEventListener('scroll', () => {
            logAutoScroll = body.scrollTop + body.clientHeight >= body.scrollHeight - 30;
        });
    }
});

function clearLogs() {
    const body = document.getElementById('log-body');
    if (body) body.innerHTML = '';
    const modal = document.getElementById('terminal-logs-body');
    if (modal) modal.innerHTML = '<div class="log-line">Terminal cleared.</div>';
}

function clearTrainLogs() {
    const body = document.getElementById('train-log-body');
    if (body) body.innerHTML = '<span style="color: var(--text-muted);">[YOLO Retraining Terminal Cleared]</span>';
    const modal = document.getElementById('terminal-train-body');
    if (modal) modal.innerHTML = '<span style="color: var(--text-muted);">[YOLO Retraining Terminal Cleared]</span>';
}

// ── Terminal Modal Controls ──────────────────────────────────
function openTerminalModal(type) {
    const modalId = type === 'train' ? 'terminal-train-modal' : 'terminal-logs-modal';
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.add('active');
}

function closeTerminalModal(type) {
    const modalId = type === 'train' ? 'terminal-train-modal' : 'terminal-logs-modal';
    const modal = document.getElementById(modalId);
    if (modal) modal.classList.remove('active');
}

// ── Registry Modal Controls ──────────────────────────────────
function openRegistryModal() {
    const modal = document.getElementById('registry-modal');
    if (modal) modal.classList.add('active');
    // Reload model versions into the expanded registry grid
    loadModelVersions();
}

function closeRegistryModal() {
    const modal = document.getElementById('registry-modal');
    if (modal) modal.classList.remove('active');
}

// Close modals on backdrop click
document.addEventListener('DOMContentLoaded', () => {
    ['terminal-logs-modal', 'terminal-train-modal', 'registry-modal'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', (e) => {
            if (e.target === el) el.classList.remove('active');
        });
    });
});

// ── Pipeline Collapse/Expand Controls ───────────────────────
let pipelineExpanded = true;
const PIPELINE_STEPS = [
    'yolo_inference', 'vlm_annotation', 'crop_extraction', 'feature_extraction',
    'faiss_search', 'hdbscan_cluster', 'save_manifest', 'llm_advisor', 'train_yolo'
];
const PIPELINE_STEP_NAMES = [
    'YOLO Inference', 'VLM Annotation', 'Crop Extraction', 'DINOv2 Features',
    'FAISS Search', 'HDBSCAN Clustering', 'Save Manifest', 'LLM Advisor', 'YOLO Fine-tuning'
];

function togglePipelineView() {
    pipelineExpanded = !pipelineExpanded;
    const btn = document.getElementById('btn-toggle-pipeline');
    const flow = document.getElementById('pipe-flow');
    const tray = document.getElementById('pipeline-tray');

    if (pipelineExpanded) {
        btn.textContent = 'Collapse ▬';
        flow.style.display = 'block';
        tray.style.display = 'none';
    } else {
        btn.textContent = 'Expand ✚';
        flow.style.display = 'none';
        tray.style.display = 'flex';
        updatePipelineTray();
    }
}

function updatePipelineTray() {
    let activeIdx = -1;
    let doneCount = 0;
    PIPELINE_STEPS.forEach((step, i) => {
        const el = document.getElementById(`pn-${step}`);
        if (el && el.classList.contains('pipe-node--active') && activeIdx === -1) activeIdx = i;
        if (el && el.classList.contains('pipe-node--done')) doneCount++;
    });

    let stageName = 'Idle';
    let percentage = 0;

    if (activeIdx !== -1) {
        stageName = PIPELINE_STEP_NAMES[activeIdx];
        percentage = Math.round((activeIdx / PIPELINE_STEPS.length) * 100);
    } else if (doneCount > 0) {
        if (doneCount >= PIPELINE_STEPS.length) {
            stageName = 'All Complete ✓';
            percentage = 100;
        } else {
            stageName = `Done: ${PIPELINE_STEP_NAMES[doneCount - 1]}`;
            percentage = Math.round((doneCount / PIPELINE_STEPS.length) * 100);
        }
    }

    const trayLabel = document.getElementById('pipeline-tray-current');
    const trayBar = document.getElementById('pipeline-tray-progress');
    const trayPct = document.getElementById('pipeline-tray-percentage');
    if (trayLabel) trayLabel.textContent = stageName;
    if (trayBar) trayBar.style.width = `${percentage}%`;
    if (trayPct) trayPct.textContent = `${percentage}%`;
}

// ── Drag and Drop Upload Support ─────────────────────────────
function handleDropEvent(e) {
    e.preventDefault();
    e.stopPropagation();
    const zone = document.getElementById('drop-zone');
    if (zone) zone.classList.remove('dragover');
    const files = e.dataTransfer ? e.dataTransfer.files : null;
    if (files && files.length > 0) {
        // Simulate the file input change event
        const mockEvent = { target: { files } };
        handleImageUpload(mockEvent);
    }
}

// ── Pipeline Status from Logs ───────────────────────────────
const SOURCE_TO_NODE_ID = {
    'yolo_inference': 'yolo_inference',
    'vlm_annotation': 'vlm_annotation',
    'crop_extraction': 'crop_extraction',
    'feature_extraction': 'feature_extraction',
    'faiss_search': 'faiss_search',
    'hdbscan_cluster': 'hdbscan_cluster',
    'label_studio_sync': 'save_manifest',
    'manifest_save': 'save_manifest',
    'llm_advisor': 'llm_advisor',
    'yolo_train': 'train_yolo',
    'train_yolo': 'train_yolo'
};

const NODE_NAMES = Object.keys(SOURCE_TO_NODE_ID);

let pipelineRunning = false;

function updatePipelineFromLog(evt) {
    const source = evt.source;

    // Update global status
    if (evt.message.includes('pipeline triggered') || evt.message.includes('pipeline started')) {
        setGlobalStatus('running', 'Running');
        setDiscoveryStatus('running', 'Running');
        pipelineRunning = true;
        hasDiscoveryRunThisSession = true;
        // Reset all nodes
        Object.values(SOURCE_TO_NODE_ID).forEach(nodeId => setNodeState(nodeId, ''));
        // Clear cluster display immediately to prevent showing stale data
        clearClusters();
    }

    if (evt.message.includes('pipeline finished') || evt.message.includes('pipeline complete')) {
        setGlobalStatus('complete', 'Complete');
        setDiscoveryStatus('complete', 'Complete');
        document.getElementById('btn-discover').disabled = false;
        pipelineRunning = false;
        loadStats();
    }

    if (evt.message.includes('pipeline failed')) {
        setGlobalStatus('failed', 'Failed');
        setDiscoveryStatus('failed', 'Failed');
        document.getElementById('btn-discover').disabled = false;
        pipelineRunning = false;
    }

    // Update individual nodes
    if (NODE_NAMES.includes(source)) {
        const nodeId = SOURCE_TO_NODE_ID[source];
        if (evt.level === 'step' && (evt.message.includes('Starting') || evt.message.includes('running') || evt.message.includes('Triggering'))) {
            setNodeState(nodeId, 'active');
        }
        if (evt.level === 'info' && (evt.message.includes('complete') || evt.message.includes('saved') || evt.message.includes('finished') || evt.message.includes('deployed') || evt.message.includes('Ready for review'))) {
            setNodeState(nodeId, 'done');
            // Show cached indicator
            if (evt.data && evt.data.cached) {
                const detail = document.getElementById(`pd-${nodeId}`);
                if (detail) detail.textContent = `${evt.data.items_processed} items (cached)`;
            } else if (evt.data && evt.data.items_processed !== undefined) {
                const detail = document.getElementById(`pd-${nodeId}`);
                if (detail) detail.textContent = `${evt.data.items_processed} items`;
            }
        }
        if (evt.level === 'error') {
            setNodeState(nodeId, 'fail');
        }
    }
}

function setNodeState(name, state) {
    const el = document.getElementById(`pn-${name}`);
    if (!el) return;
    el.classList.remove('pipe-node--active', 'pipe-node--done', 'pipe-node--fail');
    if (state) el.classList.add(`pipe-node--${state}`);
    // Also update the minimized tray if collapsed
    if (!pipelineExpanded) updatePipelineTray();
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

    const useCache = document.getElementById('in-use-cache').checked;

    try {
        hasDiscoveryRunThisSession = true;
        await post('/api/discovery/trigger', {
            use_sample_run: false,
            use_cache: useCache,
        });
        toast(`Pipeline started${useCache ? ' (cache mode)' : ''}`, 'success');
    } catch (e) {
        toast(`Failed: ${e.message}`, 'error');
        btn.disabled = false;
    }
}

// ── Data Loading ────────────────────────────────────────────
async function loadStats() {
    try {
        const [cfg, clusters, versions] = await Promise.all([
            get('/api/config'),
            get('/api/clusters'),
            get('/api/model-versions'),
        ]);

        document.getElementById('s-images').textContent = cfg.input_images_count || '0';
        document.getElementById('s-clusters').textContent = (clusters.clusters || []).length || '0';
        document.getElementById('s-models').textContent = (versions.versions || []).length || '0';

        const known = cfg.known_defect_names || [];
        document.getElementById('s-known').textContent = known.length || '0';
        document.getElementById('cfg-known-defects').textContent = known.length ? known.join(', ') : 'None';

        // Only render clusters if pipeline isn't running and discovery has run in this session
        if (!clusters.pipeline_running) {
            if (hasDiscoveryRunThisSession) {
                renderClusters(clusters.clusters || []);
            } else {
                renderClusters([]);
            }
        }
    } catch (e) {
        console.error('Stats load failed:', e);
    }
}

function clearClusters() {
    const grid = document.getElementById('clusters-grid');
    grid.innerHTML = `
        <div class="empty" style="grid-column:1/-1">
            <div class="empty-icon">⏳</div>
            <div>Pipeline running — clusters will appear when ready...</div>
        </div>`;
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
        
        // Trigger Retraining Advisor panel immediately
        showLLMAdvisorRetrainWorkflow();
    } catch (e) {
        toast(`Failed to save names: ${e.message}`, 'error');
    }
}

// ── Status Polling (lightweight, supplements SSE) ───────────
async function pollStatus() {
    try {
        const s = await get('/api/status');
        // Update discovery button
        if (s.discovery) {
            if (s.discovery.status === 'running') {
                hasDiscoveryRunThisSession = true;
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
    <div class="batch-bar" id="batch-bar" style="display:none; justify-content: space-between; align-items: center; padding: 12px; background: var(--bg-elevated); backdrop-filter: blur(10px); border-radius: 8px; margin-bottom: 16px; border: 1px solid var(--border-medium);">
      <div style="color:var(--text-primary); font-weight: 500; font-size:14px;"><span id="batch-count">0</span> crops selected</div>
      <div style="display:flex; gap:8px; align-items:center;">
        <span style="font-size:13px; color:var(--text-secondary);">Move to:</span>
        <select id="batch-target-select" style="padding:6px 12px; border-radius:4px; background:var(--bg-surface); color:var(--text-primary); border:1px solid var(--border-medium); cursor:pointer; font-size:13px;">
          ${batchOptions}
        </select>
        <button class="btn btn--primary btn--sm" onclick="runBatchMove('${clusterName}')" style="padding: 6px 12px; font-size: 13px;">Apply Move</button>
        <button class="btn btn--danger btn--sm" onclick="runBatchDrop('${clusterName}')" style="padding: 6px 12px; font-size: 13px; background: var(--error);">🗑️ Drop</button>
      </div>
    </div>

    <!-- Select All Controls -->
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; padding:0 4px;">
        <label style="display:flex; align-items:center; gap:8px; font-size:13px; font-weight:600; cursor:pointer; color:var(--text-primary); user-select:none;">
            <input type="checkbox" id="select-all-crops" onchange="toggleSelectAllCrops('${clusterName}', this.checked)" style="width:16px; height:16px; accent-color:var(--success); cursor:pointer;" />
            Select All
        </label>
        <span style="font-size:12px; color:var(--text-muted); font-style:italic;">Check items to apply batch operations</span>
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
        // Reload stats to refresh cluster data, then reopen modal
        await loadStats();
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
        await loadStats();
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
        // Reload stats then reopen modal to show it's gone
        await loadStats();
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
        // Reload stats then reopen modal
        await loadStats();
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

// ── Model Management ────────────────────────────────────────

// Floating tooltip (JS-driven, reliable)
function showFloatingTooltip(el) {
    const tip = document.getElementById('floating-tooltip');
    if (!tip) return;
    const raw = el.dataset.tip || '';
    tip.innerHTML = raw.replace(/\n/g, '<br>');
    tip.style.display = 'block';
    const rect = el.getBoundingClientRect();
    tip.style.left = '-9999px';
    tip.style.top  = '-9999px';
    // Wait for size
    requestAnimationFrame(() => {
        const tipH = tip.offsetHeight;
        const tipW = tip.offsetWidth;
        let top  = rect.top - tipH - 10 + window.scrollY;
        let left = rect.left + rect.width / 2 - tipW / 2;
        if (top < 8) top = rect.bottom + 8 + window.scrollY;
        if (left < 8) left = 8;
        if (left + tipW > window.innerWidth - 8) left = window.innerWidth - tipW - 8;
        tip.style.top  = top  + 'px';
        tip.style.left = left + 'px';
    });
}
function hideFloatingTooltip() {
    const tip = document.getElementById('floating-tooltip');
    if (tip) tip.style.display = 'none';
}


async function loadModelVersions() {

    try {
        const data = await get('/api/model-versions');
        const versions = data.versions || [];
        const current = data.current;

        // Update active model pill
        const pill = document.getElementById('model-active-pill');
        const text = document.getElementById('model-active-text');
        if (current) {
            pill.className = 'status-pill status-pill--complete';
            text.textContent = current;
        } else {
            pill.className = 'status-pill status-pill--idle';
            text.textContent = 'No active model';
        }

        // Compact registry view — show icon only, JS floating tooltip on hover
        const activeVersion = versions.find(v => v.version_id === current);
        const compactVersion = document.getElementById('lbl-compact-model-version');
        const compactDate = document.getElementById('lbl-compact-model-date');
        const trigger = document.getElementById('compact-model-classes-trigger');

        if (activeVersion) {
            if (compactVersion) compactVersion.textContent = activeVersion.version_id;
            if (compactDate) compactDate.textContent = activeVersion.created_at
                ? new Date(activeVersion.created_at).toLocaleString()
                : (activeVersion.timestamp || 'Unknown date');
            const classes = activeVersion.classes || [];
            if (trigger) {
                if (classes.length) {
                    trigger.style.display = 'inline-flex';
                    trigger.innerHTML = `<span
                        data-tip="Classes (${classes.length})\n${classes.join('\n')}"
                        onmouseenter="showFloatingTooltip(this)"
                        onmouseleave="hideFloatingTooltip()"
                        style="cursor:help;font-size:0.85rem;padding:3px 10px;border-radius:4px;background:rgba(255,255,255,0.06);border:1px solid #3f3f46;color:#a1a1aa;display:inline-flex;align-items:center;gap:5px;">
                        🏷️ <span style="font-size:0.75rem;">${classes.length} classes</span>
                    </span>`;
                } else {
                    trigger.style.display = 'none';
                }
            }
        } else {
            if (compactVersion) compactVersion.textContent = 'No active model';
            if (compactDate) compactDate.textContent = 'Run the pipeline to register a model';
            if (trigger) trigger.style.display = 'none';
        }

        document.getElementById('s-models').textContent = versions.length || '0';
        renderModelVersions(versions, current);
    } catch (e) {
        console.error('Model versions load failed:', e);
    }
}

function renderModelVersions(versions, current) {
    const grid = document.getElementById('model-versions-grid');
    if (!grid) return;
    if (!versions.length) {
        grid.innerHTML = `
            <div class="empty" style="grid-column:1/-1">
                <div class="empty-icon">📦</div>
                <div>No model versions registered yet</div>
            </div>`;
        return;
    }

    let html = '';
    for (const v of versions) {
        const isActive = v.version_id === current;
        const metrics = v.metrics || {};
        const classes = v.classes || [];
        const created = v.created_at ? new Date(v.created_at).toLocaleString() : v.timestamp || '—';

        // Floating tooltip texts via data-tip attribute (avoids HTML injection)
        const classesTip = classes.length
            ? `Classes (${classes.length})\n${classes.join('\n')}`
            : '';
        const metricsTip = Object.keys(metrics).length
            ? 'Metrics\n' + Object.entries(metrics).map(([k, val]) => {
                const d = typeof val === 'number' ? (val * 100).toFixed(1) + '%' : val;
                return k + ': ' + d;
              }).join('\n')
            : '';

        const mkIcon = (emoji, tip, label) => tip
            ? `<span data-tip="${escapeHtml(tip)}" onmouseenter="showFloatingTooltip(this)" onmouseleave="hideFloatingTooltip()" style="cursor:help;font-size:0.82rem;padding:3px 10px;border-radius:4px;background:rgba(255,255,255,0.05);border:1px solid #3f3f46;color:#a1a1aa;display:inline-flex;align-items:center;gap:4px;">${emoji} <span>${label}</span></span>`
            : '';

        // Action buttons
        let actionsHtml = '<div class="model-card-actions">';
        if (!isActive) {
            actionsHtml += `<button class="btn btn--success btn--sm" onclick="deployModel('${v.version_id}')">🚀 Deploy</button>`;
            actionsHtml += `<button class="btn btn--outline btn--sm" onclick="rollbackModel('${v.version_id}')">↩️ Rollback</button>`;
        } else {
            actionsHtml += `<span class="status-pill status-pill--complete" style="font-size:11px;">✓ Active</span>`;
        }
        actionsHtml += `<button class="btn btn--outline btn--sm" onclick="showModelDetails('${v.version_id}')">📋 Details</button>`;
        actionsHtml += '</div>';

        const iconBar = mkIcon('🏷️', classesTip, classes.length + ' classes') +
                         mkIcon('📊', metricsTip, 'metrics');


        html += `
        <div class="model-card${isActive ? ' model-card--active' : ''}">
            <div class="model-card-title">${escapeHtml(v.version_id)}</div>
            <div class="model-card-meta">
                <span>📅 ${created}</span>
                <span>💾 ${v.size_mb || '?'} MB</span>
                <span>🔧 ${v.source || 'unknown'}</span>
            </div>
            <div style="display:flex;gap:8px;margin-top:8px;align-items:center;flex-wrap:wrap;">${iconBar}</div>
            ${actionsHtml}
        </div>`;
    }

    grid.innerHTML = html;
}

async function deployModel(versionId) {
    if (!confirm(`Deploy model version ${versionId} as the active model?`)) return;
    try {
        toast(`Deploying ${versionId}...`, 'info');
        const r = await post('/api/models/deploy-confirm', { version_id: versionId });
        if (r.success) {
            toast(`Model ${versionId} deployed successfully!`, 'success');
            loadModelVersions();
        } else {
            toast(`Deploy failed: ${r.error}`, 'error');
        }
    } catch (e) {
        toast(`Deploy failed: ${e.message}`, 'error');
    }
}

async function rollbackModel(versionId) {
    if (!confirm(`Rollback to model version ${versionId}? This will replace the current active model and rebuild FAISS.`)) return;
    try {
        toast(`Rolling back to ${versionId}...`, 'info');
        const r = await post('/api/models/rollback', { version_id: versionId });
        if (r.success) {
            toast(`Rolled back to ${versionId} (FAISS: ${r.faiss_vectors} vectors)`, 'success');
            loadModelVersions();
        } else {
            toast(`Rollback failed: ${r.error}`, 'error');
        }
    } catch (e) {
        toast(`Rollback failed: ${e.message}`, 'error');
    }
}

function showModelDetails(versionId) {
    // Find version from loaded data
    get('/api/model-versions').then(data => {
        const v = (data.versions || []).find(x => x.version_id === versionId);
        if (!v) { toast('Version not found', 'error'); return; }

        const modal = document.getElementById('model-modal');
        const title = document.getElementById('model-modal-title');
        const body = document.getElementById('model-modal-body');

        title.textContent = `Model: ${v.version_id}`;

        let html = `
        <div class="advisor-section">
            <div class="advisor-section-title">📋 General Info</div>
            <div class="advisor-params">
                <span class="advisor-param-key">Version:</span>
                <span class="advisor-param-val">${v.version_id}</span>
                <span class="advisor-param-key">Created:</span>
                <span class="advisor-param-val">${v.created_at || v.timestamp || '—'}</span>
                <span class="advisor-param-key">Size:</span>
                <span class="advisor-param-val">${v.size_mb || '?'} MB</span>
                <span class="advisor-param-key">Source:</span>
                <span class="advisor-param-val">${v.source || 'unknown'}</span>
                <span class="advisor-param-key">Status:</span>
                <span class="advisor-param-val">${v.status || '—'}</span>
            </div>
        </div>`;

        // Metrics
        const metrics = v.metrics || {};
        if (Object.keys(metrics).length) {
            html += `<div class="advisor-section">
                <div class="advisor-section-title">📊 Metrics</div>
                <div class="advisor-params">`;
            for (const [k, val] of Object.entries(metrics)) {
                const display = typeof val === 'number' ? (val * 100).toFixed(1) + '%' : val;
                html += `<span class="advisor-param-key">${k}:</span>
                         <span class="advisor-param-val">${display}</span>`;
            }
            html += '</div></div>';
        }

        // Training config
        const cfg = v.training_config || {};
        if (Object.keys(cfg).length) {
            html += `<div class="advisor-section">
                <div class="advisor-section-title">⚙️ Training Config</div>
                <div class="advisor-params">`;
            for (const [k, val] of Object.entries(cfg)) {
                html += `<span class="advisor-param-key">${k}:</span>
                         <span class="advisor-param-val">${val}</span>`;
            }
            html += '</div></div>';
        }

        // Notes
        if (v.notes) {
            html += `<div class="advisor-section">
                <div class="advisor-section-title">📝 Notes</div>
                <p style="font-size:0.8rem; color:var(--text-secondary);">${escapeHtml(v.notes)}</p>
            </div>`;
        }

        body.innerHTML = html;
        modal.classList.add('active');
    });
}

function closeModelModal() {
    document.getElementById('model-modal').classList.remove('active');
}


// ── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
    loadStats();
    loadModelVersions();
    loadRecentLogs();
    startLogStream();
    pollStatus();

    // Cache vs Upload row interactions
    const cacheCheckbox = document.getElementById('in-use-cache');
    if (cacheCheckbox) {
        cacheCheckbox.addEventListener('change', (e) => {
            const uploadRow = document.getElementById('upload-row');
            if (e.target.checked) {
                uploadRow.style.opacity = '0.4';
                uploadRow.style.pointerEvents = 'none';
                document.getElementById('upload-status').textContent = 'Using cached annotations (upload skipped)';
            } else {
                uploadRow.style.opacity = '1';
                uploadRow.style.pointerEvents = 'auto';
                document.getElementById('upload-status').textContent = 'Using default input dataset (80 images)';
                document.getElementById('upload-input').value = '';
            }
        });
    }

    // Periodic stats refresh (every 15s)
    setInterval(loadStats, 15000);
    setInterval(loadModelVersions, 30000);
    setInterval(pollStatus, 5000);
});

// ── Upload & Camera Helpers ──────────────────────────────────
async function handleImageUpload(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const statusEl = document.getElementById('upload-status');
    statusEl.innerHTML = `⏳ Uploading ${files.length} images...`;
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    try {
        const response = await fetch('/api/images/upload', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error(response.statusText);
        const result = await response.json();
        statusEl.innerHTML = `✅ Loaded ${result.count} images successfully`;
        toast(`Uploaded ${result.count} images for pipeline run`, 'success');
        
        // Refresh input image count stat
        loadStats();
    } catch (e) {
        statusEl.innerHTML = `❌ Upload failed: ${e.message}`;
        toast(`Upload failed: ${e.message}`, 'error');
    }
}

function triggerCameraFeedDemo() {
    toast('Live Metrology Camera Feed: Offline (Demo Mode — Standby)', 'warning');
}


// ── MLOps Retraining Console Workflow ───────────────────────
let activeAdvisorConfig = null;

async function showLLMAdvisorRetrainWorkflow() {
    switchRightTab('train');
    const card = document.getElementById('train-advisor-card');
    const content = document.getElementById('train-advisor-content');
    
    if (!card || !content) return;
    
    card.style.display = 'block';
    card.style.animation = 'pulse 1.5s 2';
    content.innerHTML = `<span style="color:var(--text-secondary);">Querying Gemini Retraining Advisor...</span>`;
    
    try {
        const data = await get('/api/retraining/advisor-preview');
        const rec = data.recommendation || {};
        const meta = data.metadata || {};
        
        activeAdvisorConfig = rec.config || { epochs: 30, imgsz: 640, batch_size: 16 };
        
        content.innerHTML = `
            <div style="font-weight:bold; margin-bottom: 6px; color: ${rec.should_train ? 'var(--success)' : 'var(--error)'};">
                ${rec.should_train ? '✅ Recommended to Train' : '⚠️ Not Recommended to Train'}
            </div>
            <div style="font-size:12px; margin-bottom:8px;">${escapeHtml(rec.reason || '')}</div>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; background:rgba(0,0,0,0.03); padding:8px; border-radius:6px; border:1px solid var(--border); font-size:11px;">
                <div>📦 Classes: <strong>${meta.num_classes || 0}</strong></div>
                <div>🖼️ Total Crops: <strong>${meta.total_crops || 0}</strong></div>
                <div>⏱️ Epochs: <strong>${activeAdvisorConfig.epochs || activeAdvisorConfig.epochs || 30}</strong></div>
                <div>📐 Image Size: <strong>${activeAdvisorConfig.imgsz || 640}</strong></div>
            </div>
        `;
    } catch (e) {
        content.innerHTML = `<span style="color:var(--error);">Failed to load advisor recommendation: ${escapeHtml(e.message)}</span>`;
    }
}

async function acceptAdvisorRecommendation() {
    const card = document.getElementById('train-advisor-card');
    if (card) card.style.display = 'none';
    
    toast('Starting retraining with Advisor parameters...', 'info');
    
    const params = activeAdvisorConfig || { epochs: 30, imgsz: 640, batch_size: 16 };
    try {
        const body = {
            epochs: params.epochs || 30,
            imgsz: params.imgsz || 640,
            batch_size: params.batch || params.batch_size || 16
        };
        await post('/api/retraining/smart-trigger', body);
        toast('YOLO fine-tuning started!', 'success');
    } catch (e) {
        toast(`Training failed to start: ${e.message}`, 'error');
        if (card) card.style.display = 'block';
    }
}

function openCustomizeModal() {
    const modal = document.getElementById('model-modal');
    const title = document.getElementById('model-modal-title');
    const body = document.getElementById('model-modal-body');
    
    title.textContent = '⚙️ Customize YOLO Retraining Parameters';
    
    body.innerHTML = `
    <div style="display:flex; flex-direction:column; gap:12px; font-size:14px; color:var(--text-primary);">
        <div class="input-group">
            <label style="font-weight:bold; display:block; margin-bottom:4px;">Epochs</label>
            <input type="number" id="cust-epochs" value="30" min="1" max="300" style="width:100%; padding:8px; border-radius:4px; border:1px solid var(--border); background:var(--bg-secondary); color:var(--text-primary);" />
        </div>
        <div class="input-group">
            <label style="font-weight:bold; display:block; margin-bottom:4px;">Image Size (imgsz)</label>
            <input type="number" id="cust-imgsz" value="640" min="32" max="1280" style="width:100%; padding:8px; border-radius:4px; border:1px solid var(--border); background:var(--bg-secondary); color:var(--text-primary);" />
        </div>
        <div class="input-group">
            <label style="font-weight:bold; display:block; margin-bottom:4px;">Batch Size</label>
            <input type="number" id="cust-batch" value="16" min="1" max="128" style="width:100%; padding:8px; border-radius:4px; border:1px solid var(--border); background:var(--bg-secondary); color:var(--text-primary);" />
        </div>
        <div class="flex gap-sm" style="margin-top:12px;">
            <button class="btn btn--primary" onclick="submitCustomRetraining()">🚀 Start Retraining</button>
            <button class="btn btn--outline" onclick="closeModelModal()">Cancel</button>
        </div>
    </div>
    `;
    modal.classList.add('active');
}

async function submitCustomRetraining() {
    const epochs = parseInt(document.getElementById('cust-epochs').value) || 30;
    const imgsz = parseInt(document.getElementById('cust-imgsz').value) || 640;
    const batch = parseInt(document.getElementById('cust-batch').value) || 16;
    
    closeModelModal();
    const card = document.getElementById('train-advisor-card');
    if (card) card.style.display = 'none';
    
    toast(`Starting custom training (Epochs: ${epochs}, Image Size: ${imgsz}, Batch: ${batch})...`, 'info');
    
    try {
        await post('/api/retraining/smart-trigger', {
            epochs: epochs,
            imgsz: imgsz,
            batch_size: batch
        });
        toast('Custom YOLO fine-tuning triggered!', 'success');
    } catch (e) {
        toast(`Training failed: ${e.message}`, 'error');
        if (card) card.style.display = 'block';
    }
}

async function triggerFactoryReset() {
    if (!confirm('⚠️ SYSTEM FACTORY RESET WARNING:\n\nThis will completely reset the DYN-EYE system back to its initial state:\n- Swaps active model back to YOLO v1.\n- Sets known defect categories to 6 initial classes.\n- Deletes all fine-tuned model versions and history.\n- Clears all crop & cluster directories.\n- Rebuilds a clean FAISS index from the 6 initial classes.\n\nAre you absolutely sure you want to proceed?')) {
        return;
    }
    try {
        toast('Factory resetting system...', 'info');
        const r = await post('/api/system/reset-all', {});
        if (r.success) {
            toast('System factory reset successfully!', 'success');
            document.getElementById('log-body').innerHTML = '';
            document.getElementById('train-log-body').innerHTML = '<span style="color: var(--text-muted);">[YOLO Retraining Terminal Idle]</span>';
            const tLogsBody = document.getElementById('terminal-logs-body');
            const tTrainBody = document.getElementById('terminal-train-body');
            if (tLogsBody) tLogsBody.innerHTML = '<div class="log-line">System reset. Waiting for pipeline events...</div>';
            if (tTrainBody) tTrainBody.innerHTML = '<span style="color: var(--text-muted);">[Retraining Terminal Idle]</span>';
            document.getElementById('val-train-epoch').textContent = '—';
            document.getElementById('val-train-loss').textContent = '—';
            // Reset pipeline tray too
            const trayLabel = document.getElementById('pipeline-tray-current');
            const trayBar = document.getElementById('pipeline-tray-progress');
            const trayPct = document.getElementById('pipeline-tray-percentage');
            if (trayLabel) trayLabel.textContent = 'Idle';
            if (trayBar) trayBar.style.width = '0%';
            if (trayPct) trayPct.textContent = '0%';
            // Reload all data
            hasDiscoveryRunThisSession = false;
            await Promise.all([loadStats(), loadModelVersions()]);
        }
    } catch (e) {
        toast(`Reset failed: ${e.message}`, 'error');
    }
}

// ── Select All Cluster Crops ─────────────────────────────────
function toggleSelectAllCrops(clusterName, isChecked) {
    const checkboxes = document.querySelectorAll('.crop-checkbox');
    checkboxes.forEach(cb => { cb.checked = isChecked; });
    updateBatchUI(clusterName);
}

