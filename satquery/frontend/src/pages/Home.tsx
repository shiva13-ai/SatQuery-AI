/* Orbital Observatory style: neo-instrumental mission control, signal orange actions, quiet depth, asymmetric chat + scope composition. */
import { FormEvent, useEffect, useRef, useState } from "react";
import {
  Activity,
  Archive,
  BarChart3,
  Bot,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Clock3,
  Command,
  Copy,
  Database,
  FileImage,
  FolderClock,
  ImagePlus,
  Layers3,
  Link2,
  Loader2,
  MapPinned,
  Menu,
  MoreHorizontal,
  Paperclip,
  PanelLeft,
  PanelRight,
  Plus,
  Radar,
  Search,
  Send,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  SquarePen,
  Trash2,
  UploadCloud,
  Waves,
  X,
  Zap,
  ZoomIn,
} from "lucide-react";
import { toast } from "sonner";

const ASSETS = {
  mark: "/satellite-lens-mark.png",
  orbital: "/orbital-observatory-reference.png",
  delta: "/coastal-delta-capture.png",
  thermal: "/urban-thermal-capture.png",
  radar: "/spectral-radar-texture.png",
  video: "/satellite_animation.webm",
};

type Mode = "single" | "bi-temporal" | "optical-sar";
const MAX_CAPTURE_BYTES = 12 * 1024 * 1024;

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1] ?? "");
    reader.onerror = () => reject(reader.error ?? new Error("Could not read file"));
    reader.readAsDataURL(file);
  });
}

function uploadCaptureWithProgress(
  payload: { fileName: string; mimeType: string; sizeBytes: number; dataBase64: string },
  onProgress: (percent: number) => void,
): Promise<{ fileUrl: string }> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/captures/upload");
    request.setRequestHeader("Content-Type", "application/json");
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.min(95, Math.round((event.loaded / event.total) * 100)));
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) {
        try {
          resolve(JSON.parse(request.responseText) as { fileUrl: string });
        } catch {
          reject(new Error("Storage response was invalid"));
        }
      } else {
        reject(new Error(`Capture upload failed (${request.status})`));
      }
    };
    request.onerror = () => reject(new Error("Capture upload network error"));
    request.send(JSON.stringify(payload));
  });
}

type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  time: string;
  rich?: boolean;
};

const MODES: Array<{
  id: Mode;
  label: string;
  note: string;
  icon: typeof ImagePlus;
}> = [
  { id: "single", label: "Single Image", note: "Inspect one capture", icon: ImagePlus },
  { id: "bi-temporal", label: "Bi-Temporal Pair", note: "Compare two dates", icon: Layers3 },
  { id: "optical-sar", label: "Optical + SAR Pair", note: "Fuse visual + radar", icon: MapPinned },
];

const initialMessages: Message[] = [
  {
    id: "prompt-1",
    role: "user",
    text: "Compare these two captures and trace the meaningful surface change. Focus on built-up area, vegetation stress, and anything that looks like flood movement.",
    time: "09:41",
  },
  {
    id: "answer-1",
    role: "assistant",
    text: "I compared both scenes across the built-up edge, vegetation response, and low-lying channels. The strongest signal is a new, compact footprint on the east bank with a corresponding moisture shift in the adjacent plots.",
    time: "09:42",
    rich: true,
  },
];

function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className={compact ? "brand brand-compact" : "brand"}>
      <span className="brand-mark-wrap">
        <span className="mark-orbit-symbol" aria-hidden="true"><span /></span>
        <img src={ASSETS.mark} alt="" className="brand-mark" />
      </span>
      {!compact && (
        <span className="brand-wordmark">SATELLITE <b>LENS</b></span>
      )}
    </div>
  );
}

function ModeMenu({ mode, onChange }: { mode: Mode; onChange: (mode: Mode) => void }) {
  const [open, setOpen] = useState(false);
  const selected = MODES.find((item) => item.id === mode) ?? MODES[0];
  const SelectedIcon = selected.icon;

  return (
    <div className="mode-picker">
      <button
        type="button"
        className="mode-picker-trigger"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <SelectedIcon size={16} strokeWidth={1.7} />
        <span>{selected.label}</span>
        <ChevronDown size={14} className={open ? "rotate-180" : ""} />
      </button>
      {open && (
        <div className="mode-menu" role="menu">
          <div className="mode-menu-label">ANALYSIS INPUT</div>
          {MODES.map((item) => {
            const Icon = item.icon;
            const isSelected = item.id === mode;
            return (
              <button
                type="button"
                role="menuitem"
                key={item.id}
                className={`mode-option ${isSelected ? "is-selected" : ""}`}
                onClick={() => {
                  onChange(item.id);
                  setOpen(false);
                }}
              >
                <span className="mode-option-icon"><Icon size={18} strokeWidth={1.5} /></span>
                <span className="mode-option-copy">
                  <strong>{item.label}</strong>
                  <small>{item.note}</small>
                </span>
                {isSelected && <Check size={15} className="mode-check" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function OrbitalVisual() {
  return (
    <div className="orbital-visual" aria-hidden="true">
      <img src={ASSETS.orbital} alt="" className="orbital-reference" />
      <div className="orbital-grid" />
      <div className="orbit orbit-one" />
      <div className="orbit orbit-two" />
      <div className="orbit orbit-three" />
      <div className="orbital-core">
        <div className="core-latitude core-latitude-one" />
        <div className="core-latitude core-latitude-two" />
        <div className="core-meridian" />
        <div className="core-glint" />
      </div>
      <span className="orbital-node node-one" />
      <span className="orbital-node node-two" />
      <span className="orbital-node node-three" />
      <span className="orbital-label label-one">01 / ORBITAL</span>
      <span className="orbital-label label-two">SIGNAL READY</span>
    </div>
  );
}

function Sidebar({ onNewAnalysis }: { onNewAnalysis: () => void }) {
  const [active, setActive] = useState("New analysis");
  const navItems = [
    { label: "New analysis", icon: SquarePen },
    { label: "Scene library", icon: Radar },
    { label: "Saved findings", icon: BarChart3 },
  ];
  const utilityItems = [
    { label: "Capture archive", icon: Archive },
    { label: "Data sources", icon: Database },
  ];

  const clickNav = (label: string) => {
    setActive(label);
    if (label === "New analysis") onNewAnalysis();
    else toast(`${label} is ready for the next build phase.`);
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-topline">
        <Logo />
        <button className="icon-button sidebar-collapse" type="button" aria-label="Collapse mission rail" onClick={() => toast("Mission rail stays pinned in this prototype.")}>
          <PanelLeft size={16} />
        </button>
      </div>

      <button className="new-analysis-button" type="button" onClick={onNewAnalysis}>
        <span className="new-analysis-icon"><Plus size={16} /></span>
        <span>New analysis</span>
        <kbd><Command size={11} /> K</kbd>
      </button>

      <div className="sidebar-section">
        <div className="sidebar-label">WORKSPACE</div>
        <nav className="sidebar-nav">
          {navItems.map(({ label, icon: Icon }) => (
            <button key={label} type="button" className={`sidebar-nav-item ${active === label ? "is-active" : ""}`} onClick={() => clickNav(label)}>
              <Icon size={16} strokeWidth={1.65} />
              <span>{label}</span>
              {label === "New analysis" && <span className="active-dot" />}
            </button>
          ))}
        </nav>
      </div>

      <div className="sidebar-section sidebar-section-spaced">
        <div className="sidebar-label">RECENT PASSES <span>04</span></div>
        <div className="recent-list">
          <button type="button" className="recent-item is-current" onClick={() => toast("Current pass already open.")}>
            <span className="recent-swatch swatch-orange" />
            <span className="recent-copy"><strong>Coastal delta</strong><small>Change scan · 09:42</small></span>
            <ChevronRight size={14} />
          </button>
          <button type="button" className="recent-item" onClick={() => toast("Pass loaded as a placeholder.")}>
            <span className="recent-swatch swatch-mint" />
            <span className="recent-copy"><strong>Urban fringe</strong><small>Thermal read · 08:16</small></span>
            <ChevronRight size={14} />
          </button>
          <button type="button" className="recent-item" onClick={() => toast("Pass loaded as a placeholder.")}>
            <span className="recent-swatch swatch-blue" />
            <span className="recent-copy"><strong>River corridor</strong><small>Flood trace · Tue</small></span>
            <ChevronRight size={14} />
          </button>
        </div>
      </div>

      <div className="sidebar-section sidebar-section-spaced utility-section">
        <div className="sidebar-label">LIBRARY</div>
        <nav className="sidebar-nav">
          {utilityItems.map(({ label, icon: Icon }) => (
            <button key={label} type="button" className="sidebar-nav-item" onClick={() => toast(`${label} is ready for the next build phase.`)}>
              <Icon size={16} strokeWidth={1.65} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
      </div>

      <div className="sidebar-footer">
        <div className="system-status"><span className="status-pulse" /><span>Lens engine online</span><span className="mono">v0.8.4</span></div>
        <div className="profile-row">
          <div className="profile-avatar">AK</div>
          <div className="profile-copy"><strong>Aarav K.</strong><small>Student workspace</small></div>
          <button type="button" className="icon-button" aria-label="Open settings" onClick={() => toast("Workspace settings are coming soon.")}><Settings2 size={16} /></button>
        </div>
      </div>
    </aside>
  );
}

function UserMessage({ message }: { message: Message }) {
  return (
    <div className="message-block user-message-block">
      <div className="message-meta"><span className="message-role">You</span><span className="message-time">{message.time}</span></div>
      <div className="user-message-bubble">{message.text}</div>
    </div>
  );
}

function AssistantMessage({ message, onCopy }: { message: Message; onCopy: () => void }) {
  return (
    <div className="message-block assistant-message-block">
      <div className="message-meta assistant-meta">
        <div className="assistant-identity"><span className="assistant-avatar"><Bot size={15} /></span><span className="message-role">Lens engine</span><span className="status-badge"><span /> source-bound</span></div>
        <div className="message-meta-actions"><span className="message-time">{message.time}</span><button type="button" className="message-action" aria-label="Copy response" onClick={onCopy}><Copy size={14} /></button><button type="button" className="message-action" aria-label="More response actions" onClick={() => toast("Response actions are coming soon.")}><MoreHorizontal size={14} /></button></div>
      </div>
      <div className="assistant-message-copy"><p>{message.text}</p></div>
      {message.rich ? <RichAnalysis onCopy={onCopy} /> : <div className="assistant-followup"><span className="signal-tick" /> Ready for a new scene or follow-up question.</div>}
    </div>
  );
}

function RichAnalysis({ onCopy }: { onCopy: () => void }) {
  return (
    <div className="rich-analysis">
      <div className="finding-grid">
        <div className="finding-card finding-card-hot">
          <div className="finding-card-top"><span className="finding-index">01</span><span className="finding-confidence">0.92 CONFIDENCE</span></div>
          <strong>New built-up footprint</strong>
          <p>Approx. 1.8 ha appears on the east bank between captures.</p>
          <div className="finding-meter"><span style={{ width: "92%" }} /></div>
        </div>
        <div className="finding-card">
          <div className="finding-card-top"><span className="finding-index">02</span><span className="finding-confidence">0.78 CONFIDENCE</span></div>
          <strong>Vegetation stress</strong>
          <p>NDVI drop clusters around the access road and field edge.</p>
          <div className="finding-meter mint"><span style={{ width: "78%" }} /></div>
        </div>
        <div className="finding-card">
          <div className="finding-card-top"><span className="finding-index">03</span><span className="finding-confidence">0.64 CONFIDENCE</span></div>
          <strong>Moisture shift</strong>
          <p>Low-lying plots show a connected radar response after rain.</p>
          <div className="finding-meter blue"><span style={{ width: "64%" }} /></div>
        </div>
      </div>
      <div className="evidence-card">
        <div className="evidence-image-wrap">
          <div className="evidence-raster evidence-raster-thermal" role="img" aria-label="False-color satellite view of an urban edge" />
          <div className="evidence-crosshair" />
          <span className="evidence-tag"><Activity size={12} /> CHANGE ZONE A</span>
          <button type="button" className="evidence-zoom" aria-label="Zoom into evidence" onClick={() => toast("Evidence viewer is coming soon.")}><ZoomIn size={14} /></button>
        </div>
        <div className="evidence-copy">
          <div className="evidence-title-row"><span className="eyebrow-label">EVIDENCE FRAME</span><span className="mono">B14 / 2 OF 2</span></div>
          <strong>Compact orange signal aligns with cleared surface</strong>
          <p>The footprint is distinct from the surrounding agricultural texture and persists across both bands.</p>
          <div className="chip-row"><span className="specimen-chip">L2A · 10 m</span><span className="specimen-chip">17 AUG 2024</span><span className="specimen-chip">31.204° N</span></div>
        </div>
      </div>
      <div className="response-actions">
        <button type="button" onClick={() => toast("Finding bookmarked.")}><span className="action-icon"><Check size={13} /></span> Bookmark finding</button>
        <button type="button" onClick={onCopy}><Copy size={14} /> Copy answer</button>
        <button type="button" onClick={() => toast("Export prepared as a placeholder.")}><Link2 size={14} /> Export</button>
      </div>
    </div>
  );
}

function SatellitePromptSurface({ locked, onToggleLock }: { locked: boolean; onToggleLock: () => void }) {
  return (
    <div className={`satellite-prompt-surface ${locked ? "is-locked" : ""}`}>
      <div className="scene-topline">
        <span className="scene-kicker"><span className="scene-live-dot" /> LIVE ORBITAL INPUT</span>
        <span className="mono">{locked ? "SCENE LOCKED" : "TRACKING / 01"}</span>
      </div>
      <div className="scene-viewport" aria-label="Animated satellite ready for image analysis" role="img">
        <div className="scene-stars"><span /><span /><span /><span /><span /><span /></div>
        <div className="scene-orbit scene-orbit-wide" />
        <div className="scene-orbit scene-orbit-tight" />
        <div className="scene-orbit scene-orbit-vertical" />
        <div className="scene-earth-glow" />
        <div className="scene-earth" />
        <div className="scene-satellite">
          <div className="satellite-boom" />
          <div className="satellite-panel satellite-panel-left"><span /><span /><span /><span /></div>
          <div className="satellite-body"><div className="satellite-window" /><div className="satellite-band" /><div className="satellite-signal" /></div>
          <div className="satellite-panel satellite-panel-right"><span /><span /><span /><span /></div>
          <div className="satellite-antenna"><span /></div>
        </div>
        <div className="scene-readout"><span className="readout-label">ACTIVE SENSOR</span><strong>ORBITAL LENS / 01</strong><span>{locked ? "Target aligned · ready to trace" : "Awaiting capture or pair"}</span></div>
        <button type="button" className="scene-lock-button" onClick={onToggleLock}><span className="scene-lock-icon">{locked ? <Check size={12} /> : <Zap size={12} />}</span>{locked ? "Scene locked" : "Lock scene"}</button>
        <div className="scene-coordinate mono">31.204° N&nbsp;&nbsp;121.472° E</div>
      </div>
      <div className="scene-bottomline"><span><span className="signal-tick" /> {locked ? "Capture lane aligned for analysis." : "Bring a capture into the orbital lane."}</span><span className="mono">3D TRACK · 60 FPS</span></div>
    </div>
  );
}

function SatelliteMotionStage() {
  return (
    <div className="satellite-motion-stage" role="img" aria-label="Animated satellite orbiting above Earth">
      <div className="motion-stars"><span /><span /><span /><span /><span /><span /><span /><span /></div>
      <div className="motion-orbit motion-orbit-wide" />
      <div className="motion-orbit motion-orbit-mid" />
      <div className="motion-orbit motion-orbit-vertical" />
      <div className="motion-earth-glow" />
      <div className="motion-earth" />
      <div className="motion-satellite">
        <div className="motion-boom" />
        <div className="motion-panel motion-panel-left"><span /><span /><span /></div>
        <div className="motion-body"><div className="motion-window" /><div className="motion-band" /></div>
        <div className="motion-panel motion-panel-right"><span /><span /><span /></div>
        <div className="motion-antenna"><span /></div>
      </div>
    </div>
  );
}

function SatelliteVideoStage() {
  return (
    <div className="satellite-video-stage">
      <video className="satellite-video" autoPlay muted loop playsInline preload="metadata" poster={ASSETS.orbital} aria-label="Satellite Lens AI visual introduction">
        <source src={ASSETS.video} type="video/webm" />
      </video>
      <div className="satellite-video-shade" />
      <div className="satellite-video-copy">
        <span className="video-kicker"><span className="scene-live-dot" /> SATELLITE-ONLY INTELLIGENCE</span>
        <h1>Satellite Lens <em>AI</em></h1>
        <p>Your imagery copilot for comparing captures, tracing surface change, and turning orbit data into clear evidence.</p>
      </div>
      <div className="satellite-video-meta"><span className="mono">REMOTE SENSING / ONLINE</span><span className="video-scan-line" /><span className="mono">LENS ENGINE 0.8</span></div>
    </div>
  );
}

function Composer({ mode, onModeChange, onSubmit }: { mode: Mode; onModeChange: (mode: Mode) => void; onSubmit: (text: string) => void }) {
  const [draft, setDraft] = useState("");
  const [fileName, setFileName] = useState("");
  const [uploadStatus, setUploadStatus] = useState<"idle" | "uploading" | "stored" | "error">("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedMode = MODES.find((item) => item.id === mode) ?? MODES[0];

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      toast("Please choose an image capture.");
      event.target.value = "";
      return;
    }
    if (file.size > MAX_CAPTURE_BYTES) {
      toast("Capture must be smaller than 12 MB.");
      event.target.value = "";
      return;
    }

    setFileName(file.name);
    setUploadProgress(0);
    setUploadStatus("uploading");
    try {
      const dataBase64 = await fileToBase64(file);
      await uploadCaptureWithProgress({ fileName: file.name, mimeType: file.type, sizeBytes: file.size, dataBase64 }, setUploadProgress);
      setUploadProgress(100);
      setUploadStatus("stored");
      toast("Capture stored securely.");
    } catch (error) {
      console.error("[Capture upload]", error);
      setUploadProgress(0);
      setUploadStatus("error");
      toast("Capture upload failed. Please try again.");
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (!draft.trim() && !fileName) {
      toast("Type a satellite question or attach a capture first.");
      return;
    }
    onSubmit(draft.trim() || `Scope the attached ${selectedMode.label.toLowerCase()} for meaningful surface change.`);
    setDraft("");
    setFileName("");
    setUploadProgress(0);
    setUploadStatus("idle");
  };

  return (
    <div className="composer-area">
      <form className="composer" onSubmit={submit}>
        <div className="composer-topline">
          <ModeMenu mode={mode} onChange={onModeChange} />
          <span className="composer-guidance"><ShieldCheck size={13} /> Images stay scoped to this analysis</span>
        </div>
        <textarea value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="Ask about the satellite image…" rows={2} aria-label="Ask about the satellite image" />
        <div className="composer-toolbar">
          <div className="composer-tools">
            <input ref={fileInputRef} type="file" accept="image/*" className="visually-hidden" onChange={handleFileChange} />
            <button type="button" className="toolbar-button attach-button" onClick={() => fileInputRef.current?.click()} disabled={uploadStatus === "uploading"}>{uploadStatus === "uploading" ? <Loader2 size={17} className="spin" /> : uploadStatus === "stored" ? <Check size={17} /> : <Paperclip size={17} />}<span>{uploadStatus === "uploading" ? `Storing capture · ${uploadProgress}%` : uploadStatus === "stored" ? "Capture stored" : "Attach capture"}</span></button>
            <button type="button" className="toolbar-button" onClick={() => toast("Layer controls are coming soon.")}><SlidersHorizontal size={16} /><span>Layers</span></button>
            {fileName && <span className={`file-chip ${uploadStatus === "error" ? "file-chip-error" : ""}`}><FileImage size={13} />{fileName}<button type="button" aria-label="Remove attached file" onClick={() => { setFileName(""); setUploadProgress(0); setUploadStatus("idle"); if (fileInputRef.current) fileInputRef.current.value = ""; }}><X size={12} /></button></span>}
            {uploadStatus === "uploading" && <span className="upload-progress" role="status" aria-live="polite"><span className="upload-progress-track"><span style={{ width: `${uploadProgress}%` }} /></span>{uploadProgress}%</span>}
          </div>
          <div className="composer-send-group"><span className="model-note">Lens 0.8 <ChevronDown size={13} /></span><button type="submit" className="send-button" aria-label="Send analysis request"><Send size={16} /></button></div>
        </div>
      </form>
      <div className="composer-footnote"><span>Satellite Lens can misread ambiguous terrain.</span><button type="button" onClick={() => toast("Feedback form is coming soon.")}>Report an issue</button></div>
    </div>
  );
}

function ScopePanel({ onClose }: { onClose: () => void }) {
  return (
    <aside className="scope-panel">
      <div className="scope-header">
        <div><div className="eyebrow-label">ANALYSIS SCOPE</div><h2>Change scan</h2></div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close analysis scope"><X size={17} /></button>
      </div>
      <div className="scope-preview">
        <div className="scope-raster" role="img" aria-label="Multispectral raster texture" />
        <div className="scope-preview-overlay" />
        <div className="scope-crosshair"><span /><span /></div>
        <div className="scope-preview-meta"><span>DELTA / EAST BANK</span><strong>+24.8%</strong></div>
        <button type="button" className="scope-expand" aria-label="Expand scope preview" onClick={() => toast("Scope preview is already focused.")}><ZoomIn size={15} /></button>
      </div>
      <div className="scope-status-row"><span className="scope-status-icon"><Check size={15} /></span><div><strong>Pass complete</strong><span>2 scenes · 3 layers · 18.4 sec</span></div><span className="mono">09:42</span></div>
      <div className="scope-divider" />
      <div className="scope-section"><div className="scope-section-head"><span className="eyebrow-label">TELEMETRY</span><button type="button" className="mini-action" onClick={() => toast("Telemetry options are coming soon.")}><MoreHorizontal size={15} /></button></div><div className="telemetry-list"><div><span>Change magnitude</span><strong className="orange-text">HIGH</strong></div><div><span>Surface class</span><strong>BUILT-UP EDGE</strong></div><div><span>Cloud cover</span><strong>2.1%</strong></div><div><span>Scene alignment</span><strong className="mint-text">98.6%</strong></div></div></div>
      <div className="scope-section"><div className="scope-section-head"><span className="eyebrow-label">CAPTURE TRAIL</span><span className="mono">BI-TEMPORAL</span></div><div className="capture-trail"><div className="trail-line" /><div className="trail-point trail-point-start"><span /><div><strong>04 AUG 2024</strong><small>Baseline · Sentinel-2</small></div></div><div className="trail-point trail-point-end"><span /><div><strong>17 AUG 2024</strong><small>Comparison · Sentinel-2</small></div></div></div></div>
      <div className="scope-section scope-location"><div className="scope-section-head"><span className="eyebrow-label">CENTER POINT</span><MapPinned size={15} /></div><div className="coordinates"><strong>31° 12′ 14.4″ N</strong><strong>121° 28′ 19.2″ E</strong></div><div className="chip-row"><span className="specimen-chip">AOI 4.7 km²</span><span className="specimen-chip">UTM 51R</span></div></div>
      <div className="scope-actions"><button type="button" className="scope-primary" onClick={() => toast("Finding bookmarked.")}><Plus size={15} /> Add to findings</button><button type="button" className="scope-secondary" onClick={() => toast("Export prepared as a placeholder.")}><UploadCloud size={15} /> Export scope</button></div>
    </aside>
  );
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("bi-temporal");
  const [messages, setMessages] = useState<Message[]>([]);
  const [thinking, setThinking] = useState(false);
  const [scopeOpen, setScopeOpen] = useState(true);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  useEffect(() => {
    if (!thinking) return;
    const timer = window.setTimeout(() => {
      setThinking(false);
      setMessages((current) => [...current, {
        id: `answer-${Date.now()}`,
        role: "assistant",
        text: "The scene is queued and ready. I found a clear visual story to follow, but I’d keep the conclusion directional until the next capture confirms the boundary.",
        time: "now",
      }]);
    }, 900);
    return () => window.clearTimeout(timer);
  }, [thinking]);

  const handleSubmit = (text: string) => {
    setMessages((current) => [...current, { id: `prompt-${Date.now()}`, role: "user", text, time: "now" }]);
    setThinking(true);
  };

  const newAnalysis = () => {
    setMessages([]);
    setThinking(false);
    toast("Fresh analysis workspace opened.");
  };

  const copyResponse = () => {
    toast("Response copied to clipboard.");
  };

  return (
    <div className={`app-shell ${scopeOpen ? "scope-is-open" : "scope-is-closed"}`}>
      <div className="ambient ambient-top" />
      <div className="ambient ambient-bottom" />
      <div className={`mobile-overlay ${mobileNavOpen ? "is-visible" : ""}`} onClick={() => setMobileNavOpen(false)} />
      <div className={`mobile-sidebar-wrap ${mobileNavOpen ? "is-visible" : ""}`}><Sidebar onNewAnalysis={newAnalysis} /></div>
      <div className="desktop-sidebar"><Sidebar onNewAnalysis={newAnalysis} /></div>

      <main className="chat-shell">
        <header className="topbar">
          <div className="topbar-left"><button type="button" className="mobile-menu-button icon-button" onClick={() => setMobileNavOpen(true)} aria-label="Open navigation"><Menu size={18} /></button><div className="breadcrumb"><span>MISSION / 04</span><ChevronRight size={13} /><strong>Coastal delta · change scan</strong></div></div>
          <div className="topbar-actions"><div className="connection-pill"><span /> local workspace</div><button type="button" className="icon-button" aria-label="Search workspace" onClick={() => toast("Workspace search is coming soon.")}><Search size={16} /></button><button type="button" className="icon-button" aria-label="Open help" onClick={() => toast("Help center is coming soon.")}><CircleHelp size={16} /></button><button type="button" className="icon-button" aria-label="More workspace actions" onClick={() => toast("Workspace actions are coming soon.")}><MoreHorizontal size={17} /></button></div>
        </header>

        {messages.length === 0 ? <div className="clean-home-stage"><SatelliteVideoStage /></div> : <div className="conversation-scroll"><div className="conversation-content">{messages.map((message) => message.role === "user" ? <UserMessage key={message.id} message={message} /> : <AssistantMessage key={message.id} message={message} onCopy={copyResponse} />)}{thinking && <div className="thinking-row"><span className="assistant-avatar"><Bot size={15} /></span><div><span className="thinking-label">Lens engine is reading the scene</span><div className="thinking-bars"><span /><span /><span /></div></div></div>}</div></div>}

        <Composer mode={mode} onModeChange={setMode} onSubmit={handleSubmit} />
      </main>

      {scopeOpen && <ScopePanel onClose={() => setScopeOpen(false)} />}
    </div>
  );
}
