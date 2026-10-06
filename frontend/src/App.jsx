import { useEffect, useRef, useState } from 'react';
import {
  API_BASE,
  getChannels,
  getHealth,
  getMetrics,
  predictEEG,
} from './api.js';

const CONFIG_OPTIONS = [
  {
    value: '22',
    label: '22 Channels',
    sub: 'Full 10-20 Reference Montage',
    tag: 'Baseline',
  },
  {
    value: '5',
    label: '5 Channels',
    sub: 'ANOVA Data-Driven Subset',
    tag: 'Top-5',
  },
  {
    value: '4',
    label: '4 Channels',
    sub: 'Compact Optimal Subset',
    tag: 'Top-4',
  },
];

const ACCEPT_TYPES = '.edf,.edf.gz';

function isValidEdf(name) {
  if (!name) return false;
  const lower = String(name).toLowerCase();
  return lower.endsWith('.edf') || lower.endsWith('.edf.gz');
}

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

function fmtMetric(value) {
  if (value === null || value === undefined) return 'NA';
  const n = Number(value);
  if (!Number.isFinite(n)) return 'NA';
  return n.toFixed(4);
}

function fmtPercent(value) {
  if (value === null || value === undefined) return null;
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return `${(n * 100).toFixed(2)}%`;
}

export default function App() {
  const [backend, setBackend] = useState({ state: 'checking', info: null });
  const [channels, setChannels] = useState(null);
  const [channelsUnavailable, setChannelsUnavailable] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [metricsUnavailable, setMetricsUnavailable] = useState(true);

  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [fileError, setFileError] = useState(null);
  const [config, setConfig] = useState('22');

  const [analyzing, setAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const inflight = useRef(false);

  useEffect(() => {
    refreshBackend();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshBackend() {
    setBackend({ state: 'checking', info: null });
    setChannels(null);
    setChannelsUnavailable(true);
    setMetrics(null);
    setMetricsUnavailable(true);
    try {
      const info = await getHealth();
      setBackend({ state: 'online', info });
      loadData();
    } catch {
      setBackend({ state: 'offline', info: null });
    }
  }

  async function loadData() {
    try {
      const ch = await getChannels();
      setChannels(ch);
      setChannelsUnavailable(!ch);
    } catch {
      setChannels(null);
      setChannelsUnavailable(true);
    }
    try {
      const m = await getMetrics();
      setMetrics(m);
      setMetricsUnavailable(!m || m.available === false);
    } catch {
      setMetrics(null);
      setMetricsUnavailable(true);
    }
  }

  function handleFile(selected) {
    setFileError(null);
    setError(null);
    setResult(null);
    if (!selected) {
      setFile(null);
      return;
    }
    if (!isValidEdf(selected.name)) {
      setFile(null);
      setFileError('Unsupported file format. Please select an EDF recording (.edf or .edf.gz).');
      return;
    }
    setFile(selected);
  }

  function onFileChange(event) {
    const selected = event.target.files && event.target.files[0];
    handleFile(selected);
  }

  function onDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }

  function onDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }

  function onDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }

  async function onAnalyze() {
    if (inflight.current) return;
    if (!file) {
      setError('Please select an EDF recording file first.');
      return;
    }
    inflight.current = true;
    setAnalyzing(true);
    setAnalysisStep(1);
    setError(null);
    setResult(null);

    // Simulate multi-stage visual loader steps for better UX
    const t1 = setTimeout(() => setAnalysisStep(2), 600);
    const t2 = setTimeout(() => setAnalysisStep(3), 1200);

    try {
      const data = await predictEEG(file, config);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed.');
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      inflight.current = false;
      setAnalyzing(false);
      setAnalysisStep(0);
    }
  }

  const online = backend.state === 'online';
  const probability = result && result.probability !== undefined ? result.probability : null;
  const probFormatted = fmtPercent(probability);
  const isSeizure = result && String(result.prediction).toLowerCase().includes('seizure') && !String(result.prediction).toLowerCase().includes('non');

  const channelsUsed = Array.isArray(result && result.channels_used)
    ? result.channels_used.join(', ')
    : null;

  return (
    <div className="app">
      {/* Top Navbar Header */}
      <header className="navbar">
        <div className="brand-group">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a10 10 0 0 0-10 10c0 4.42 2.87 8.17 6.84 9.5.5.08.66-.23.66-.5v-1.69c-2.77.6-3.36-1.34-3.36-1.34-.46-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.87 1.52 2.34 1.07 2.91.83.1-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.92 0-1.11.38-2 1.03-2.71-.1-.25-.45-1.29.1-2.64 0 0 .84-.27 2.75 1.02.79-.22 1.65-.33 2.5-.33.85 0 1.71.11 2.5.33 1.91-1.29 2.75-1.02 2.75-1.02.55 1.35.2 2.39.1 2.64.65.71 1.03 1.6 1.03 2.71 0 3.82-2.34 4.66-4.57 4.91.36.31.69.92.69 1.85V21c0 .27.16.59.67.5C19.14 20.16 22 16.42 22 12A10 10 0 0 0 12 2z"/>
              <path d="M4 12h4l2-4 3 8 2-6 2 2h3" />
            </svg>
          </div>
          <div className="title-area">
            <h1>NeuroSelect</h1>
            <p>
              <span>EEG Seizure Prediction Engine</span>
              <span className="model-tag">CNN-BiLSTM</span>
            </p>
          </div>
        </div>

        <div className="status-pill">
          <span className={`status-dot ${backend.state}`} />
          {backend.state === 'checking' && <span>Connecting to Backend...</span>}
          {backend.state === 'online' && <span>System Online</span>}
          {backend.state === 'offline' && <span>Backend Offline</span>}
          <span className="api-address">({API_BASE})</span>
          {!online && backend.state === 'offline' && (
            <button type="button" className="btn-retry" onClick={refreshBackend}>
              Retry
            </button>
          )}
        </div>
      </header>

      {error && (
        <div className="alert-banner" role="alert" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => setError(null)}
            style={{ background: 'none', border: 'none', color: '#fecdd3', cursor: 'pointer', fontSize: '16px', fontWeight: 'bold', padding: '0 4px' }}
            title="Dismiss notification"
          >
            ✕
          </button>
        </div>
      )}

      {/* Main Dashboard Content */}
      <main className="dashboard-grid">
        {/* Left Column: Upload & Config */}
        <section className="card card-upload">
          <div className="card-title">
            <span>EEG File Upload & Setup</span>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
          </div>

          {!file ? (
            <div
              className={`dropzone ${dragActive ? 'active' : ''}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
            >
              <input
                type="file"
                accept={ACCEPT_TYPES}
                onChange={onFileChange}
                disabled={analyzing}
                className="file-input-hidden"
              />
              <div className="dropzone-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/>
                </svg>
              </div>
              <p className="dropzone-text">
                Drag & Drop your EEG recording file here, or <span className="dropzone-highlight">Browse</span>
              </p>
              <span className="file-format-hint" style={{ fontSize: '11px', color: '#64748b', marginTop: '6px', display: 'block' }}>
                Supports standard European Data Format (.edf, .edf.gz)
              </span>
            </div>
          ) : (
            <div>
              <div className="selected-file-card">
                <div className="file-info">
                  <div className="file-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
                    </svg>
                  </div>
                  <div className="file-details">
                    <div className="name">{file.name}</div>
                    <div className="size">{formatBytes(file.size)}</div>
                  </div>
                </div>
                <button
                  type="button"
                  className="btn-remove-file"
                  onClick={() => setFile(null)}
                  disabled={analyzing}
                  title="Remove file"
                >
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>

              {/* Dynamic Waveform Visual Feedback */}
              <div className="eeg-preview-box">
                <div className="eeg-preview-header">
                  <span>RECORDING SIGNAL PREVIEW</span>
                  <span>256 Hz | RAW EEG</span>
                </div>
                <svg className="eeg-waveform-svg" viewBox="0 0 500 40" preserveAspectRatio="none">
                  <path
                    d="M0 20 Q 25 5, 50 20 T 100 20 T 150 35 T 200 10 T 250 20 T 300 25 T 350 15 T 400 30 T 450 10 T 500 20"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="1.5"
                    opacity="0.85"
                  />
                  <path
                    d="M0 20 Q 20 30, 60 15 T 120 25 T 180 10 T 240 30 T 300 15 T 360 20 T 420 35 T 480 15 T 500 20"
                    fill="none"
                    stroke="#8b5cf6"
                    strokeWidth="1"
                    opacity="0.5"
                  />
                </svg>
              </div>
            </div>
          )}

          {fileError && <p className="field-error" style={{ color: '#f43f5e', fontSize: '13px', marginTop: '8px' }}>{fileError}</p>}

          {/* Configuration Selection */}
          <div className="config-section-title">Electrode Configuration</div>
          <div className="config-grid">
            {CONFIG_OPTIONS.map((opt) => (
              <div
                key={opt.value}
                className={`config-tile ${config === opt.value ? 'selected' : ''}`}
                onClick={() => {
                  if (!analyzing) {
                    setConfig(opt.value);
                    setError(null);
                  }
                }}
              >
                <div className="config-tile-header">
                  <span className="config-ch-count">{opt.label}</span>
                  <span className="radio-indicator" />
                </div>
                <span className="config-tile-note">{opt.sub}</span>
              </div>
            ))}
          </div>

          <button
            type="button"
            className="btn-analyze"
            onClick={onAnalyze}
            disabled={analyzing || !online || !file}
          >
            {analyzing ? (
              <>
                <span className="pulse-loader" style={{ width: '20px', height: '20px', borderWidth: '2px' }} />
                <span>Running Pipeline...</span>
              </>
            ) : (
              <>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                <span>Analyze EEG Recording</span>
              </>
            )}
          </button>
        </section>

        {/* Right Column: Prediction Results */}
        <section className="card card-result">
          <div className="card-title">
            <span>Inference Output</span>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <activity width="20" height="20" />
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>

          {!result && !analyzing && (
            <div className="result-placeholder">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
              </svg>
              <p>No analysis yet</p>
              <span style={{ fontSize: '12px' }}>Upload an EDF file and click "Analyze EEG Recording" to view model inference predictions.</span>
            </div>
          )}

          {analyzing && (
            <div className="analyzing-box">
              <div className="pulse-loader">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                </svg>
              </div>
              <div className="analyzing-text">Analyzing Recording Signals</div>
              <div className="analyzing-sub">
                {analysisStep === 1 && '▶ Preprocessing: Band-pass filter (0.5–40 Hz)...'}
                {analysisStep === 2 && '▶ Segmenting 10s sliding windows...'}
                {analysisStep === 3 && '▶ Evaluating CNN-BiLSTM Feature Representations...'}
              </div>
            </div>
          )}

          {result && (
            <div className="prediction-container">
              <div className="prediction-badge-box">
                <div className="prediction-label">Prediction Outcome</div>
                <div className={`prediction-status ${isSeizure ? 'seizure' : 'normal'}`}>
                  {isSeizure ? 'SEIZURE PREDICTED' : 'NON-SEIZURE DETECTED'}
                </div>

                {probability !== null && (
                  <div className="prob-meter-box">
                    <div className="prob-meter-labels">
                      <span>Seizure Risk Index</span>
                      <span style={{ fontWeight: '700', fontFamily: 'var(--font-mono)' }}>{probFormatted}</span>
                    </div>
                    <div className="prob-bar-track">
                      <div
                        className={`prob-bar-fill ${isSeizure ? 'seizure' : 'normal'}`}
                        style={{ width: `${Math.min(Math.max(probability * 100, 5), 100)}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              <div className="details-list">
                {result.configuration !== undefined && (
                  <div className="detail-row">
                    <span className="detail-key">Active Configuration</span>
                    <span className="detail-val">{result.configuration} Channels</span>
                  </div>
                )}
                {result.segments_analyzed !== undefined && (
                  <div className="detail-row">
                    <span className="detail-key">Segments Analyzed</span>
                    <span className="detail-val">{result.segments_analyzed} windows</span>
                  </div>
                )}
                {channelsUsed && (
                  <div className="detail-row">
                    <span className="detail-key">Channels Selected</span>
                    <span className="detail-val" style={{ fontSize: '12px' }}>{channelsUsed}</span>
                  </div>
                )}
                <div className="detail-row">
                  <span className="detail-key">Neural Model</span>
                  <span className="detail-val">{result.model || 'CNN-BiLSTM'}</span>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Research Comparison Table */}
        <section className="card card-full">
          <div className="card-title">
            <span>22 vs 5 vs 4 — Research Benchmark Matrix</span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 'normal' }}>Validation Metrics on Holdout Split</span>
          </div>

          {metricsUnavailable ? (
            <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '14px' }}>
              <p style={{ margin: 0 }}>Metrics unavailable in current environment state.</p>
              <span style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
                Run Phase 6 training (<code style={{ color: 'var(--accent-cyan)' }}>scripts/run_experiment.py</code>) after adding dataset recordings to populate validation benchmark metrics.
              </span>
            </div>
          ) : (
            <div className="table-container">
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Configuration</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Specificity</th>
                    <th>F1 Score</th>
                    <th>ROC-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.rows.map((row) => (
                    <tr key={String(row.configuration)}>
                      <td>
                        <span className="config-badge">{row.configuration} Channels</span>
                      </td>
                      <td>{fmtMetric(row.accuracy)}</td>
                      <td>{fmtMetric(row.precision)}</td>
                      <td>{fmtMetric(row.recall)}</td>
                      <td>{fmtMetric(row.specificity)}</td>
                      <td>{fmtMetric(row.f1)}</td>
                      <td>{fmtMetric(row.roc_auc)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Selected Electrodes Explorer */}
        <section className="card card-full">
          <div className="card-title">
            <span>Optimal Electrode Selections</span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontWeight: 'normal' }}>ANOVA Feature Discriminability Ranking</span>
          </div>

          {channelsUnavailable ? (
            <div style={{ padding: '16px', color: 'var(--text-muted)', fontSize: '14px' }}>
              Phase 4 channel rankings have not been executed yet (<code style={{ color: 'var(--accent-cyan)' }}>selected_channels.json</code> absent).
            </div>
          ) : (
            <div className="channel-grid">
              <div className="channel-card">
                <div className="channel-card-title">
                  <span className="config-badge">5 Channels</span>
                  <span>Selected Top-5 Montage</span>
                </div>
                {channels && channels.top5 && channels.top5.length > 0 ? (
                  <div className="electrode-tags">
                    {channels.top5.map((name) => (
                      <span key={name} className="electrode-tag">{name}</span>
                    ))}
                  </div>
                ) : (
                  <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Awaiting electrode selection run.</p>
                )}
              </div>

              <div className="channel-card">
                <div className="channel-card-title">
                  <span className="config-badge" style={{ background: 'rgba(139, 92, 246, 0.15)', color: '#c084fc', borderColor: 'rgba(139, 92, 246, 0.3)' }}>
                    4 Channels
                  </span>
                  <span>Selected Top-4 Montage</span>
                </div>
                {channels && channels.top4 && channels.top4.length > 0 ? (
                  <div className="electrode-tags">
                    {channels.top4.map((name) => (
                      <span key={name} className="electrode-tag">{name}</span>
                    ))}
                  </div>
                ) : (
                  <p style={{ color: 'var(--text-muted)', fontSize: '13px' }}>Awaiting electrode selection run.</p>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <p>NeuroSelect AI Research Prototype • Leakage-Safe Real-Data EEG Pipeline • Not a Clinical Diagnostic Device</p>
      </footer>
    </div>
  );
}