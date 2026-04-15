/* ════════════════════════════════════════════════
   SENTINEL ALPHA — Core System Logic
   ════════════════════════════════════════════════ */

const Sentinel = {
  streamActive: false,
  statusInterval: null,
  liveInterval: null,
  notificationPermissionAsked: false,
  alertCooldownMs: 8000,
  alertHistory: new Map(),
  lastKnownRoiCount: 0,

  async pollStatus() {
    try {
      const res = await fetch('/api/status');
      const d = await res.json();
      const latency = d.edge_mode?.avg_latency_ms ?? '—';
      const mode = (d.edge_mode?.current_mode || 'standard').toUpperCase();
      const edgeLabel = document.getElementById('metricEdge');
      const latencyLabel = document.getElementById('metricLatency');
      const fpsLabel = document.getElementById('metricFPS');
      const roiCountLabel = document.getElementById('metricRoiCount');
      if (edgeLabel) edgeLabel.textContent = mode;
      if (latencyLabel) latencyLabel.textContent = latency;
      if (roiCountLabel) roiCountLabel.textContent = d.roi_zones ?? 0;
      if (fpsLabel) {
        const ms = Number(d.edge_mode?.last_latency_ms || 0);
        fpsLabel.textContent = ms > 0 ? Math.max(1, Math.round(1000 / ms)).toString() : '—';
      }
      
      // If webcam is active but streamActive is false, auto-resume (e.g. on page refresh)
      if (d.webcam_active && !this.streamActive) {
        this.streamActive = true;
        this.resumeUI();
      }

      if (typeof d.roi_zones === 'number' && d.roi_zones !== this.lastKnownRoiCount) {
        this.lastKnownRoiCount = d.roi_zones;
        await this.loadROI();
      }
      this.updateROIIndicators();

    } catch(e) { console.error("Status Poll Error", e); }
  },

  async pollLiveDetections() {
    if (!this.streamActive) return;
    try {
      const res = await fetch('/api/live_detections');
      const dets = await res.json();
      this.processDetectionAlerts(dets);
      this.renderLiveDetections(dets);
    } catch(e) {}
  },

  getNotificationPermission() {
    if (!('Notification' in window)) return 'unsupported';
    return Notification.permission;
  },

  updateNotificationBadge() {
    const badge = document.getElementById('notificationModeBadge');
    if (!badge) return;

    const permission = this.getNotificationPermission();
    badge.className = 'text-[10px] font-mono px-2 py-0.5 rounded-full border';

    if (permission === 'granted') {
      badge.textContent = 'DESKTOP ALERTS';
      badge.classList.add('bg-emerald-500/10', 'text-emerald-400', 'border-emerald-500/20');
      return;
    }

    if (permission === 'denied') {
      badge.textContent = 'TOAST ONLY';
      badge.classList.add('bg-amber-500/10', 'text-amber-300', 'border-amber-500/20');
      return;
    }

    if (permission === 'default') {
      badge.textContent = 'ALERTS READY';
      badge.classList.add('bg-blue-500/10', 'text-blue-300', 'border-blue-500/20');
      return;
    }

    badge.textContent = 'IN-APP ALERTS';
    badge.classList.add('bg-slate-800', 'text-slate-400', 'border-slate-700');
  },

  async ensureNotificationPermission() {
    if (!('Notification' in window)) {
      this.updateNotificationBadge();
      return 'unsupported';
    }

    if (Notification.permission !== 'default') {
      this.updateNotificationBadge();
      return Notification.permission;
    }

    if (this.notificationPermissionAsked) {
      this.updateNotificationBadge();
      return Notification.permission;
    }

    this.notificationPermissionAsked = true;
    try {
      const permission = await Notification.requestPermission();
      this.updateNotificationBadge();
      return permission;
    } catch (e) {
      console.error('Notification permission error', e);
      this.updateNotificationBadge();
      return Notification.permission;
    }
  },

  shouldNotifyForDetection(det) {
    if (!det || !det.alerted) return false;
    const region = det.in_roi ? 'roi' : 'global';
    const key = `${det.class_name}:${det.risk_level}:${region}`;
    const now = Date.now();
    const lastAt = this.alertHistory.get(key) || 0;
    if (now - lastAt < this.alertCooldownMs) return false;
    this.alertHistory.set(key, now);
    return true;
  },

  processDetectionAlerts(dets) {
    dets
      .filter(det => this.shouldNotifyForDetection(det))
      .forEach(det => {
        this.showDetectionToast(det);
        this.sendDesktopNotification(det);
      });
  },

  showDetectionToast(det) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    const accent =
      det.risk_level === 'High'
        ? 'border-red-400 bg-red-500/10'
        : det.risk_level === 'Medium'
          ? 'border-amber-400 bg-amber-500/10'
          : 'border-emerald-400 bg-emerald-500/10';

    toast.className = `pointer-events-auto min-w-[280px] max-w-sm rounded-sm border-l-4 border border-white/10 ${accent} bg-slate-950/95 px-4 py-3 shadow-2xl backdrop-blur`;
    toast.innerHTML = `
      <div class="flex items-start justify-between gap-3">
        <div>
          <p class="text-[10px] font-mono tracking-[0.18em] text-slate-400 uppercase">Weapon Alert</p>
          <p class="mt-1 text-sm font-headline font-bold uppercase text-on-surface">${det.class_name}</p>
          <p class="mt-1 text-[11px] font-mono text-slate-300">
            ${det.risk_level} risk • ${(Number(det.confidence || 0) * 100).toFixed(1)}% confidence${det.in_roi ? ' • ROI' : ''}
          </p>
        </div>
        <button type="button" class="text-slate-500 hover:text-slate-200 text-xs font-mono">X</button>
      </div>
    `;

    const closeBtn = toast.querySelector('button');
    if (closeBtn) {
      closeBtn.addEventListener('click', () => toast.remove());
    }

    container.prepend(toast);
    window.setTimeout(() => toast.remove(), 4500);
  },

  sendDesktopNotification(det) {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    if (document.visibilityState === 'visible' && document.hasFocus()) return;

    const region = det.in_roi ? 'ROI zone' : 'full frame';
    const note = new Notification(`Weapon detected: ${det.class_name}`, {
      body: `${det.risk_level} risk in ${region} at ${(Number(det.confidence || 0) * 100).toFixed(1)}% confidence`,
      tag: `weapon-alert-${det.class_name}-${det.in_roi ? 'roi' : 'global'}`,
      renotify: true,
    });
    window.setTimeout(() => note.close(), 5000);
  },

  updateROIIndicators() {
    const count = this.roiZones.length;
    const badge = document.getElementById('roiStatusBadge');
    const countLabel = document.getElementById('metricRoiCount');
    if (countLabel) countLabel.textContent = count;
    if (!badge) return;

    badge.classList.remove('text-emerald-300', 'text-yellow-300', 'text-amber-300');
    if (this.roiDrawing) {
      badge.textContent = `ROI DRAWING ${this.roiPoints.length} PTS`;
      badge.classList.add('text-yellow-300');
      return;
    }

    if (count > 0) {
      badge.textContent = `ROI ACTIVE ${count}`;
      badge.classList.add('text-emerald-300');
      return;
    }

    badge.textContent = 'ROI OFF';
    badge.classList.add('text-amber-300');
  },

  async loadROI() {
    try {
      const res = await fetch('/api/roi');
      const data = await res.json();
      this.roiZones = Array.isArray(data.zones) ? data.zones : [];
      this.lastKnownRoiCount = this.roiZones.length;
      this.drawROIZones();
      this.updateROIIndicators();
    } catch (e) {
      console.error('ROI sync failed', e);
    }
  },

  getFeedViewport() {
    const canvas = document.getElementById('roiCanvas');
    const outer = document.getElementById('streamOuter');
    const img = document.getElementById('mainFeed');
    const fallback = {
      left: 0,
      top: 0,
      width: canvas ? canvas.width : outer?.clientWidth || 0,
      height: canvas ? canvas.height : outer?.clientHeight || 0,
    };

    if (!outer || !img || !img.naturalWidth || !img.naturalHeight) {
      return fallback;
    }

    const outerW = outer.clientWidth;
    const outerH = outer.clientHeight;
    const mediaAspect = img.naturalWidth / img.naturalHeight;
    const outerAspect = outerW / outerH;

    let width = outerW;
    let height = outerH;
    if (mediaAspect > outerAspect) {
      height = outerW / mediaAspect;
    } else {
      width = outerH * mediaAspect;
    }

    return {
      left: (outerW - width) / 2,
      top: (outerH - height) / 2,
      width,
      height,
    };
  },

  toNormalizedPoint(clientX, clientY) {
    const canvas = document.getElementById('roiCanvas');
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const viewport = this.getFeedViewport();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    if (
      x < viewport.left ||
      y < viewport.top ||
      x > viewport.left + viewport.width ||
      y > viewport.top + viewport.height
    ) {
      return null;
    }

    return [
      (x - viewport.left) / Math.max(1, viewport.width),
      (y - viewport.top) / Math.max(1, viewport.height),
    ];
  },

  renderLiveDetections(dets) {
    const list = document.getElementById('detectionList');
    if (!list) return;

    const badge = document.getElementById('detectionCountBadge');
    if (badge) badge.textContent = `${dets.length} ACTIVE`;

    const totalMet = document.getElementById('metricTotal');
    if (totalMet) totalMet.textContent = dets.length;

    if (dets.length === 0) {
      const msg = this.streamActive ? 'SYSTEM_ARMED // SCANNING... NO_OBJECTS_DETECTED' : 'AWAITING_DATA_STREAM...';
      list.innerHTML = `<p class="text-center py-8 text-slate-600 font-mono text-[10px]">${msg}</p>`;
      return;
    }

    list.innerHTML = dets.map(d => {
      const riskColor = d.risk_level === 'High'
        ? 'text-red-300 border-red-500/30 bg-red-500/10'
        : (d.risk_level === 'Medium'
          ? 'text-amber-300 border-amber-500/30 bg-amber-500/10'
          : 'text-emerald-300 border-emerald-500/20 bg-emerald-500/10');
      const modelClass = d.model_class ? String(d.model_class) : 'mapped';
      const scope = d.in_roi ? 'ROI' : 'GLOBAL';
      const logState = d.logged ? 'LOGGED' : 'BUFFERED';
      
      return `
        <div class="bg-surface-container-high p-4 rounded-sm border border-white/10 group cursor-pointer hover:bg-surface-variant transition-colors">
          <div class="flex justify-between items-start mb-2">
            <div>
              <h4 class="text-[12px] font-mono font-bold text-emerald-300 uppercase">${d.class_name}</h4>
              <p class="mt-1 text-[9px] text-slate-500 font-mono uppercase tracking-[0.14em]">ID ${String(d.detection_id || '').slice(-8)} • ${scope}</p>
            </div>
            <div class="px-2 py-1 rounded-full border ${riskColor}">
              <span class="text-[8px] font-mono font-bold uppercase">${d.risk_level} Risk</span>
            </div>
          </div>
          <div class="grid grid-cols-2 gap-2 text-[9px] font-mono text-slate-400">
            <span>${(d.confidence * 100).toFixed(1)}% CONF.</span>
            <span class="text-right">${logState}</span>
            <span>${modelClass}</span>
            <span class="text-right uppercase">${d.source_mode || 'live'}</span>
          </div>
        </div>
      `;
    }).join('');
  },

  resumeUI() {
    const icon = document.getElementById('toggleIcon');
    const label = document.getElementById('streamStatusLabel');
    const img = document.getElementById('mainFeed');
    const placeholder = document.getElementById('feedPlaceholder');
    
    if (img) {
      img.src = '/stream?' + Date.now();
      img.onload = () => this.drawROIZones();
      img.classList.remove('hidden');
      if (placeholder) placeholder.classList.add('hidden');
      if (icon) icon.textContent = 'pause_circle';
      if (label) {
          label.textContent = 'REC [●] LIVE';
          label.classList.add('text-emerald-500');
          label.classList.remove('text-slate-500');
      }
      if (!this.liveInterval) {
        this.liveInterval = setInterval(() => this.pollLiveDetections(), 500);
      }
      this.loadROI();
    }
  },

  async toggleStream() {
    const img = document.getElementById('mainFeed');
    const placeholder = document.getElementById('feedPlaceholder');
    const icon = document.getElementById('toggleIcon');
    const label = document.getElementById('streamStatusLabel');
    const canvas = document.getElementById('roiCanvas');
    const hint = document.getElementById('roiHint');

    if (!this.streamActive) {
      try {
        await this.ensureNotificationPermission();
        await fetch('/stream/start', { method: 'POST' });
        this.streamActive = true;
        this.resumeUI();
      } catch(e) { console.error(e); }
    } else {
      try {
        await fetch('/stream/stop', { method: 'POST' });
        this.streamActive = false;
        if (img) {
            img.src = '';
            img.classList.add('hidden');
        }
        if (placeholder) placeholder.classList.remove('hidden');
        if (icon) icon.textContent = 'play_circle';
        if (label) {
            label.textContent = 'REC [OFF] STANDBY';
            label.classList.remove('text-emerald-500');
            label.classList.add('text-slate-500');
        }
        
        clearInterval(this.liveInterval);
        this.liveInterval = null;
        this.alertHistory.clear();
        this.roiPoints = [];
        this.roiDrawing = false;
        if (canvas) {
          canvas.classList.add('hidden');
          canvas.style.pointerEvents = 'none';
        }
        if (hint) hint.classList.add('hidden');
        this.renderLiveDetections([]);
        this.updateROIIndicators();
      } catch(e) { console.error(e); }
    }
  },

  roiDrawing: false,
  roiPoints: [],
  roiZones: [],

  setupROI() {
    const canvas = document.getElementById('roiCanvas');
    const outer = document.getElementById('streamOuter');
    if (!canvas || !outer) return;

    const resize = () => {
      canvas.width = outer.clientWidth;
      canvas.height = outer.clientHeight;
      this.drawROIZones();
    };
    resize();
    window.addEventListener('resize', resize);
    canvas.style.pointerEvents = 'none';

    canvas.onclick = (e) => {
      if (!this.roiDrawing) return;
      const point = this.toNormalizedPoint(e.clientX, e.clientY);
      if (!point) return;
      this.roiPoints.push(point);
      this.drawROIZones();
      this.updateROIIndicators();
    };

    canvas.oncontextmenu = (e) => {
      if (!this.roiDrawing) return;
      e.preventDefault();
      this.sealROIZone();
    };
  },

  drawROIZones() {
    const canvas = document.getElementById('roiCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const viewport = this.getFeedViewport();

    // Existing zones
    this.roiZones.forEach(z => {
      ctx.beginPath();
      ctx.strokeStyle = '#4edea3';
      ctx.fillStyle = 'rgba(78, 222, 163, 0.1)';
      z.forEach((p, i) => {
        const x = viewport.left + (p[0] * viewport.width);
        const y = viewport.top + (p[1] * viewport.height);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.fill();
    });

    // Current zone points
    if (this.roiPoints.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = '#fbbf24';
      this.roiPoints.forEach((p, i) => {
        const x = viewport.left + (p[0] * viewport.width);
        const y = viewport.top + (p[1] * viewport.height);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        ctx.moveTo(x + 3, y);
        ctx.arc(x, y, 3, 0, Math.PI * 2);
      });
      ctx.stroke();
    }
  },

  async sealROIZone() {
    if (this.roiPoints.length < 3) {
      this.roiPoints = [];
      this.drawROIZones();
      if (this.roiDrawing) this.toggleROIDrawing(false);
      return;
    }
    this.roiZones.push([...this.roiPoints]);
    this.roiPoints = [];
    this.drawROIZones();
    const res = await fetch('/set_roi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ zones: this.roiZones })
    });
    const data = await res.json();
    this.roiZones = Array.isArray(data.zones) ? data.zones : this.roiZones;
    this.lastKnownRoiCount = this.roiZones.length;
    this.updateROIIndicators();
    this.toggleROIDrawing(false);
  },

  toggleROIDrawing(force) {
    this.roiDrawing = force !== undefined ? force : !this.roiDrawing;
    const canvas = document.getElementById('roiCanvas');
    const hint = document.getElementById('roiHint');
    const btn = document.getElementById('toggleROI');
    if (!canvas || !hint || !btn) return;
    
    if (this.roiDrawing) {
      canvas.classList.remove('hidden');
      canvas.style.pointerEvents = 'auto';
      hint.classList.remove('hidden');
      btn.textContent = 'SEAL_ZONE';
      btn.classList.add('text-yellow-400');
      this.drawROIZones();
    } else {
      if (this.roiPoints.length > 0) this.sealROIZone();
      canvas.classList.add('hidden');
      canvas.style.pointerEvents = 'none';
      hint.classList.add('hidden');
      btn.textContent = 'DRAW_ROI';
      btn.classList.remove('text-yellow-400');
    }
    this.updateROIIndicators();
  },

  async clearROI() {
    this.roiZones = [];
    this.roiPoints = [];
    this.drawROIZones();
    await fetch('/clear_roi', { method: 'POST' });
    this.lastKnownRoiCount = 0;
    this.toggleROIDrawing(false);
    this.updateROIIndicators();
  },

  init() {
    this.statusInterval = setInterval(() => this.pollStatus(), 3000);
    this.pollStatus();
    this.updateNotificationBadge();
    this.loadROI();

    // Wire up Live page
    const toggleBtn = document.getElementById('toggleStream');
    if (toggleBtn) toggleBtn.onclick = () => this.toggleStream();

    const clearBtn = document.getElementById('clearROI');
    if (clearBtn) clearBtn.onclick = () => this.clearROI();

    const roiBtn = document.getElementById('toggleROI');
    if (roiBtn) roiBtn.onclick = () => this.toggleROIDrawing();

    this.setupROI();
  }
};

document.addEventListener('DOMContentLoaded', () => Sentinel.init());
