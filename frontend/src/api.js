// NeuroSelect frontend API module.
// All backend calls live here; components never talk to fetch directly.
// The endpoints implement the documented Phase 8 / Phase 9 contract:
//
//   GET  /health         -> {"status":"ok","model_ready":bool,
//                            "channels_ready":bool,"metrics_ready":bool}
//   GET  /channels       -> {"available":true,"top_5":[...],"top_4":[...]}
//   GET  /metrics        -> {"available":true,"rows":[{configuration,channels,
//                            accuracy,precision,recall,specificity,f1,
//                            roc_auc,test_samples}, ...]}
//   POST /predict        -> multipart/form-data: file (EDF), configuration
//                            ("22" | "5" | "4")
//                            -> {"prediction":"seizure"|"non_seizure",
//                                "probability":<0..1>,"segments_analyzed":int,
//                                "channels_used":[...],"configuration":"5",
//                                "model":"CNN-BiLSTM"}
//
// Nothing here is ever replaced with fake data. When the backend is
// unavailable, an ApiError is thrown and callers show that state honestly.

export const API_BASE = (
  import.meta.env?.VITE_API_URL?.replace(/\/+$/, '') || 'http://localhost:8000'
);

export class ApiError extends Error {
  constructor(kind, message) {
    super(message);
    this.kind = kind; // 'network' | 'http'
    this.name = 'ApiError';
  }
}

// FastAPI reports errors as {"detail": "..."} or {"detail": [{...,"msg":...}]}.
function extractDetail(payload) {
  if (!payload || typeof payload !== 'object') return null;
  const detail = payload.detail;
  if (typeof detail === 'string' && detail.trim()) return detail.trim();
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0];
    if (first && typeof first.msg === 'string') return first.msg;
  }
  return null;
}

async function request(path, options = {}, timeoutMs = 30000) {
  let response;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
    });
  } catch (err) {
    if (err && err.name === 'AbortError') {
      throw new ApiError('network', 'Request timed out. The analysis is taking longer than expected — please try again.');
    }
    throw new ApiError('network', 'Cannot reach backend. Make sure the backend server is running on port 8000.');
  } finally {
    clearTimeout(timer);
  }

  let payload = null;
  const text = await response.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (!response.ok) {
    throw new ApiError('http', extractDetail(payload)
      || `Request failed (HTTP ${response.status}).`);
  }
  return payload;
}

export async function getHealth() {
  const data = (await request('/health')) || {};
  return {
    status: data.status ?? 'ok',
    modelReady: Boolean(data.model_ready),
    channelsReady: Boolean(data.channels_ready),
    metricsReady: Boolean(data.metrics_ready),
  };
}

export async function getChannels() {
  const data = await request('/channels');
  if (!data || data.available === false) return null;
  return {
    top5: Array.isArray(data.top_5) ? data.top_5 : null,
    top4: Array.isArray(data.top_4) ? data.top_4 : null,
  };
}

export async function getMetrics() {
  const data = await request('/metrics');
  if (!data || data.available === false) return { available: false, rows: [] };
  const rows = Array.isArray(data)
    ? data
    : (Array.isArray(data.rows) ? data.rows : []);
  return { available: true, rows };
}

export function predictEEG(file, configuration) {
  const body = new FormData();
  body.append('file', file);
  body.append('configuration', configuration);
  // Large EDF files + TF inference can take 60-120s — use a 5 min timeout.
  return request('/predict', { method: 'POST', body }, 5 * 60 * 1000);
}