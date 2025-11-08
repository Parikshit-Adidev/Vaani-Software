# main.py
"""
Vaani — Streamlit app with improved inline recorder (Web Audio API) + TFLite inference
Place your edge_impulse_model.tflite (optional) and labels.txt (optional) in the same repo.

Run locally:
    pip install -r requirements.txt
    streamlit run main.py
"""
import streamlit as st
import numpy as np
import io, os, base64, json
import librosa
import matplotlib.pyplot as plt

# Prefer soundfile if available, but fallback to librosa if not.
try:
    import soundfile as sf
    SOUND_FILE_AVAILABLE = True
except Exception:
    sf = None
    SOUND_FILE_AVAILABLE = False

# Try TFLite interpreter
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    TFLITE_SOURCE = "tflite-runtime"
except Exception:
    try:
        from tensorflow.lite import Interpreter as TFLiteInterpreter
        TFLITE_SOURCE = "tensorflow.lite"
    except Exception:
        TFLiteInterpreter = None
        TFLITE_SOURCE = None

# Optional Vosk
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False

# -----------------------------
# Config
# -----------------------------
EI_MODEL_PATH = "edge_impulse_model.tflite"
LABELS_PATH = "labels.txt"
TARGET_SR = 16000

st.set_page_config(page_title="Vaani", layout="centered", initial_sidebar_state="collapsed")
st.title("Vaani")

# Improved frontend: uses Web Audio API ScriptProcessor to capture PCM and AnalyserNode for waveform
FRONTEND_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Vaani — Recorder (Improved)</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
:root{--bg1:#fff17a;--bg2:#ffefb3;--accent:#072b34;--glass:rgba(255,255,255,0.95);}
html,body{margin:0;height:100%;font-family:Inter,system-ui,Segoe UI,Roboto;background:linear-gradient(180deg,var(--bg1),var(--bg2));color:var(--accent)}
.container{max-width:1000px;margin:28px auto;padding:18px}
.hero{text-align:center;margin-bottom:12px}
.hero h1{margin:6px 0;font-size:32px}
.card{background:var(--glass);border-radius:14px;padding:16px;box-shadow:0 14px 36px rgba(3,27,34,0.06);border:1px solid rgba(3,27,34,0.04)}
.rec-row{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.rec-btn{width:84px;height:84px;border-radius:50%;border:none;background:linear-gradient(180deg,#ff4f4f,#ff1f1f);box-shadow:0 12px 28px rgba(255,79,79,0.22);color:white;font-weight:800;font-size:22px;cursor:pointer}
.rec-btn.recording{animation:recPulse 1.1s infinite}
@keyframes recPulse{0%{transform:scale(1)}50%{transform:scale(1.05)}100%{transform:scale(1)}}
.wavebox{flex:1;height:96px;border-radius:10px;background:linear-gradient(90deg,rgba(3,27,34,0.03),rgba(3,27,34,0.02));display:flex;align-items:center;padding:8px;border:1px dashed rgba(3,27,34,0.06)}
.canvas{width:100%;height:80px}
.controls{display:flex;flex-direction:column;gap:8px;min-width:160px;align-items:flex-end}
.btn{padding:8px 12px;border-radius:10px;border:none;background:var(--accent);color:white;font-weight:700;cursor:pointer}
.small{font-size:13px;color:rgba(11,35,40,0.75)}
textarea{width:100%;height:120px;border-radius:10px;padding:10px;font-family:monospace;font-size:13px;border:1px solid rgba(3,27,34,0.06);margin-top:12px}
.row{display:flex;gap:8px;align-items:center}
.suggests{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.chip{padding:6px 10px;border-radius:999px;background:rgba(7,59,70,0.06);cursor:pointer;border:1px solid rgba(7,59,70,0.03);font-weight:600}
@media(max-width:720px){.controls{align-items:stretch;min-width:auto}}
</style>
</head>
<body>
<div class="container">
  <div class="hero"><h1>Vaani</h1><div class="small">Improved recorder — synchronized waveform & proper WAV export</div></div>

  <div class="card">
    <h3 style="margin:0 0 8px 0">Record</h3>
    <div class="small">Speak a short sentence (3–8 s). The waveform shows actual audio level, and the resulting file is a correct WAV you can download or copy as Base64.</div>
    <div style="height:12px"></div>

    <div class="rec-row">
      <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
        <button id="recBtn" class="rec-btn">●</button>
        <div class="small">Record</div>
      </div>

      <div class="wavebox">
        <canvas id="waveCanvas" class="canvas"></canvas>
      </div>

      <div class="controls">
        <div id="timer" style="font-weight:800">00:00</div>
        <div class="row">
          <button id="stopBtn" class="btn" disabled>Stop</button>
          <button id="pauseBtn" class="btn" style="background:transparent;color:var(--accent);border:1px solid rgba(3,27,34,0.06)">Pause</button>
        </div>
      </div>
    </div>

    <div class="suggests" id="suggests">
      <div class="chip" data-sent="I have noticed a change in my voice recently.">Change in voice</div>
      <div class="chip" data-sent="I feel my speech is slower than before.">Speech slower</div>
      <div class="chip" data-sent="I have trouble pronouncing certain words.">Pronunciation</div>
      <div class="chip" data-sent="My voice is hoarse or breathy today.">Hoarseness</div>
    </div>

    <div style="margin-top:12px">
      <label style="font-weight:700">Recorded WAV (Base64)</label>
      <textarea id="b64out" placeholder="WAV base64 will appear here after you stop recording"></textarea>
      <div style="display:flex;gap:8px;margin-top:8px">
        <button id="copyBtn" class="btn">Copy Base64</button>
        <button id="downloadBtn" class="btn">Download WAV</button>
        <button id="playBtn" class="btn" style="background:transparent;color:var(--accent);border:1px solid rgba(3,27,34,0.06)">Play</button>
      </div>
      <div style="margin-top:8px" class="small">Tip: After copying the Base64, paste into the Vaani Streamlit app's paste-box and press Load.</div>
    </div>
  </div>
</div>

<script>
// Improved recorder using AudioContext + ScriptProcessor for PCM capture
let audioContext = null;
let sourceNode = null;
let processor = null;
let analyser = null;
let chunks = [];
let recording = false;
let startTime = 0;
let timerInterval = null;
let recordedBuffer = [];
let sampleRate = 44100; // will be updated to context.sampleRate
const recBtn = document.getElementById('recBtn');
const stopBtn = document.getElementById('stopBtn');
const pauseBtn = document.getElementById('pauseBtn');
const timerEl = document.getElementById('timer');
const canvas = document.getElementById('waveCanvas');
const ctx = canvas.getContext('2d');
const b64out = document.getElementById('b64out');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const playBtn = document.getElementById('playBtn');
const chips = document.querySelectorAll('.chip');

function format(ms){
  const s = Math.floor(ms/1000);
  const mm = String(Math.floor(s/60)).padStart(2,'0');
  const ss = String(s%60).padStart(2,'0');
  return mm + ":" + ss;
}

function resizeCanvas(){
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawWave(timeData){
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = 'rgba(3,27,34,0.04)';
  ctx.fillRect(0,0,w,h);
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#073b46';
  ctx.beginPath();
  const step = Math.max(1, Math.floor(timeData.length / w));
  for(let i=0; i<w; i++){
    const idx = i*step;
    const v = (timeData[idx] - 128) / 128.0; // -1..1
    const y = (h/2) + v*(h/2 - 6);
    if(i===0) ctx.moveTo(i,y); else ctx.lineTo(i,y);
  }
  ctx.stroke();
}

function startTimer(){
  startTime = Date.now();
  timerInterval = setInterval(()=> {
    timerEl.textContent = format(Date.now() - startTime);
  }, 200);
}
function stopTimer(){
  clearInterval(timerInterval);
  timerEl.textContent = "00:00";
}

// encode Float32Array -> WAV (PCM16)
function interleave(buffers, length) {
  const result = new Float32Array(length);
  let offset = 0;
  for (let i = 0; i < buffers.length; i++) {
    result.set(buffers[i], offset);
    offset += buffers[i].length;
  }
  return result;
}
function floatTo16BitPCM(float32Array) {
  const l = float32Array.length;
  const buffer = new ArrayBuffer(l * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < l; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buffer;
}
function writeWAV(float32Array, sampleRate) {
  const pcm16 = floatTo16BitPCM(float32Array);
  const buffer = new ArrayBuffer(44 + pcm16.byteLength);
  const view = new DataView(buffer);

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* file length */
  view.setUint32(4, 36 + pcm16.byteLength, true);
  /* RIFF type */
  writeString(view, 8, 'WAVE');
  /* format chunk identifier */
  writeString(view, 12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sampleRate * blockAlign) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channelCount * bytesPerSample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
  view.setUint32(40, pcm16.byteLength, true);
  // write PCM
  const pcmView = new Uint8Array(buffer, 44);
  pcmView.set(new Uint8Array(pcm16));
  return buffer;
}
function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++){
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

async function startRecording(){
  if (recording) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio:true});
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sampleRate = audioContext.sampleRate;
    sourceNode = audioContext.createMediaStreamSource(stream);

    // analyser for visual waveform
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    sourceNode.connect(analyser);

    // Buffer size 4096 for ScriptProcessor
    const bufferSize = 4096;
    processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
    const recBuffers = [];
    let recLength = 0;

    processor.onaudioprocess = function(e) {
      const input = e.inputBuffer.getChannelData(0);
      // copy buffer
      recBuffers.push(new Float32Array(input));
      recLength += input.length;
    };
    sourceNode.connect(processor);
    processor.connect(audioContext.destination); // required for processing to run in some browsers

    // visualization loop
    recording = true;
    recBtn.classList.add('recording');
    stopBtn.disabled = false;
    startTimer();

    function viz(){
      if(!recording) return;
      const data = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(data);
      drawWave(data);
      requestAnimationFrame(viz);
    }
    viz();

    // store buffers, stop handler will merge
    processor._recBuffers = recBuffers;
    processor._recLength = () => recBuffers.reduce((s,b)=>s+b.length,0);

    // attach stream to processor for later stop
    processor._stream = stream;
    // store for access
    window._vaani_processor = processor;
  } catch (err) {
    alert("Microphone permission denied or not available: " + err);
  }
}

function stopRecording(){
  if(!recording) return;
  recording = false;
  recBtn.classList.remove('recording');
  stopBtn.disabled = true;
  stopTimer();
  // get recorded buffers from processor
  const proc = window._vaani_processor;
  if(!proc){
    alert("Recording processor missing");
    return;
  }
  const recBuffers = proc._recBuffers || [];
  const length = recBuffers.reduce((s,b)=>s+b.length,0);
  const merged = interleave(recBuffers, length);
  const wavBuffer = writeWAV(merged, sampleRate);
  const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
  // base64
  const reader = new FileReader();
  reader.onloadend = function() {
    const base64data = reader.result.split(',')[1];
    b64out.value = base64data;
  };
  reader.readAsDataURL(wavBlob);
  // create download
  downloadBtn.onclick = () => {
    const url = URL.createObjectURL(wavBlob);
    const a = document.createElement('a'); a.href = url; a.download = 'vaani_recording.wav'; document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  };
  // create play
  playBtn.onclick = () => {
    const url = URL.createObjectURL(wavBlob);
    const audio = new Audio(url);
    audio.play();
  };
  // stop audio graph & stream
  try {
    proc.disconnect();
    if (proc._stream) {
      const tracks = proc._stream.getTracks();
      tracks.forEach(t=>t.stop());
    }
    if (sourceNode) sourceNode.disconnect();
    if (analyser) analyser.disconnect();
    if (audioContext) {
      // don't close to allow future reuse; but we can close and create new on next start
      audioContext.close().catch(()=>{});
    }
  } catch(e){}
  window._vaani_processor = null;
}

recBtn.addEventListener('click', ()=> {
  if(!recording) startRecording();
});

stopBtn.addEventListener('click', ()=> {
  stopRecording();
});

copyBtn.addEventListener('click', async ()=> {
  if(!b64out.value) { alert('No recording yet'); return; }
  try{
    await navigator.clipboard.writeText(b64out.value);
    alert('Base64 copied to clipboard. Paste into the Vaani Streamlit app.');
  }catch(e){
    prompt('Copy Base64:', b64out.value);
  }
});

chips.forEach(c=>c.addEventListener('click', ()=> {
  const s = c.dataset.sent || c.innerText;
  navigator.clipboard.writeText(s).then(()=> alert('Sample sentence copied to clipboard')).catch(()=> prompt('Sample sentence:', s));
}));

// pause button toggles: it will stop the graph but keep stream alive (not implemented full pause/resume)
pauseBtn.addEventListener('click', ()=>{
  alert('Pause/Resume not implemented — press Stop to end recording.');
});

// helper: ensure canvas initially has display
requestAnimationFrame(()=> drawWave(new Uint8Array(analyser?analyser.fftSize:2048).fill(128)));
</script>
</body>
</html>
"""

# Render frontend HTML in Streamlit
st.components.v1.html(FRONTEND_HTML, height=700)

st.markdown("---")
st.subheader("Paste Base64 audio from the recorder above (or upload a file)")

# Paste base64 box
b64 = st.text_area("Paste Base64 audio here", height=160)
if st.button("Load and run"):
    if not b64 or len(b64) < 50:
        st.error("Please paste a valid Base64 audio string (from the recorder) or upload a file.")
    else:
        try:
            audio_bytes = base64.b64decode(b64.strip())
            st.success("Loaded audio — playing below")
            st.audio(audio_bytes)
            st.session_state["app_audio_bytes"] = audio_bytes
        except Exception as e:
            st.error("Base64 decode failed: " + str(e))

# Upload alternative
uploaded = st.file_uploader("Or upload an audio file (wav/mp3/m4a/ogg)", type=["wav","mp3","m4a","ogg"])
if uploaded is not None:
    audio_bytes = uploaded.read()
    st.session_state["app_audio_bytes"] = audio_bytes
    st.audio(audio_bytes)

# Sidebar: model info if present
st.sidebar.title("Model & Info")
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            lbls = [l.strip() for l in f if l.strip()]
            st.sidebar.write(f"Labels: {len(lbls)}")
    except Exception:
        st.sidebar.write("labels.txt unreadable")
else:
    st.sidebar.write("No labels.txt found")

if os.path.exists(EI_MODEL_PATH):
    if TFLiteInterpreter is None:
        st.sidebar.error("No TFLite interpreter available. Add tensorflow or tflite-runtime to requirements.")
    else:
        try:
            tmp_interp = TFLiteInterpreter(model_path=EI_MODEL_PATH)
            st.sidebar.success("TFLite model file found")
            try:
                st.sidebar.write(tmp_interp.get_input_details())
            except Exception:
                pass
        except Exception as e:
            st.sidebar.error("Failed to open model: " + str(e))
else:
    st.sidebar.info("No model file (edge_impulse_model.tflite) found")

# -----------------------
# Helpers for audio & inference (same as before)
# -----------------------
def read_audio_bytes(audio_bytes: bytes):
    if SOUND_FILE_AVAILABLE:
        try:
            data, sr = sf.read(io.BytesIO(audio_bytes))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), sr
        except Exception:
            pass
    data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr

def quantize_input(arr_float: np.ndarray, input_detail: dict):
    dtype = np.dtype(input_detail['dtype'])
    q_params = input_detail.get('quantization', (0.0, 0))
    scale, zero_point = q_params
    if scale == 0 and zero_point == 0 and np.issubdtype(dtype, np.floating):
        return arr_float.astype(dtype)
    if np.issubdtype(dtype, np.integer):
        if scale == 0:
            scale = 1e-8
        arr_q = np.round(arr_float / scale + zero_point).astype(dtype)
        return arr_q
    else:
        return arr_float.astype(dtype)

def preprocess_for_model(audio_bytes: bytes, input_details):
    x, sr = read_audio_bytes(audio_bytes)
    if sr != TARGET_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    inp = input_details[0]
    shape = list(inp['shape'])
    dtype = np.dtype(inp['dtype'])
    if len(shape) == 2 or len(shape) == 1:
        n_samples = int(shape[-1])
        if x.shape[0] < n_samples:
            x_padded = np.pad(x, (0, n_samples - x.shape[0]), mode='constant')
        else:
            x_padded = x[:n_samples]
        if np.issubdtype(dtype, np.integer):
            if dtype == np.int16:
                arr = (x_padded * 32767.0).astype(np.int16)
            else:
                arr_float = x_padded.astype(np.float32)
                arr = quantize_input(arr_float, inp)
        else:
            arr = x_padded.astype(dtype)
        return arr.reshape([1, -1])
    if len(shape) == 4:
        _, frames, bins, ch = shape
        desired_frames = int(frames)
        desired_bins = int(bins)
        S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=desired_bins)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_t = S_db.T
        if S_t.shape[0] < desired_frames:
            pad_amt = desired_frames - S_t.shape[0]
            S_t = np.pad(S_t, ((0, pad_amt), (0,0)), mode='constant', constant_values=(S_t.min(),))
        else:
            S_t = S_t[:desired_frames, :]
        S_norm = (S_t - S_t.mean()) / (S_t.std() + 1e-9)
        out = S_norm.reshape((1, desired_frames, desired_bins, 1))
        if np.issubdtype(dtype, np.integer):
            out = quantize_input(out.astype(np.float32), inp)
        else:
            out = out.astype(dtype)
        return out
    # fallback
    prod = int(np.prod(shape[1:]))
    flat = x
    if flat.shape[0] < prod:
        flat = np.pad(flat, (0, prod - flat.shape[0]), mode='constant')
    else:
        flat = flat[:prod]
    flat_f = flat.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        flat_q = quantize_input(flat_f, inp)
        arr = flat_q.reshape(shape)
    else:
        arr = flat_f.reshape(shape).astype(dtype)
    return arr

def run_tflite(interpreter, input_tensor):
    indie = interpreter.get_input_details()
    outie = interpreter.get_output_details()
    interpreter.set_tensor(indie[0]['index'], input_tensor)
    interpreter.invoke()
    outputs = []
    for od in outie:
        outputs.append(interpreter.get_tensor(od['index']))
    return outputs

# -----------------------
# If audio exists in session, run inference & show results
# -----------------------
if "app_audio_bytes" in st.session_state:
    audio_bytes = st.session_state["app_audio_bytes"]
    st.markdown("## Results")
    st.audio(audio_bytes)

    # load labels
    labels = None
    if os.path.exists(LABELS_PATH):
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                labels = [l.strip() for l in f if l.strip()]
        except Exception:
            labels = None

    if os.path.exists(EI_MODEL_PATH) and TFLiteInterpreter is not None:
        try:
            interpreter = TFLiteInterpreter(model_path=EI_MODEL_PATH)
            interpreter.allocate_tensors()
            in_details = interpreter.get_input_details()
            inp_tensor = preprocess_for_model(audio_bytes, in_details)
            outputs = run_tflite(interpreter, inp_tensor)
            out = outputs[0].squeeze()
            if out.ndim > 1:
                out = out.flatten()
            try:
                if np.any(out < 0) or np.any(out > 1) or not np.isclose(out.sum(), 1.0):
                    exp = np.exp(out - np.max(out))
                    probs = exp / exp.sum()
                else:
                    probs = out.astype(np.float32)
            except Exception:
                probs = out.astype(np.float32)
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            top_label = labels[top_idx] if labels and top_idx < len(labels) else f"class_{top_idx}"
            st.markdown(f"### Prediction: **{top_label}** — confidence `{top_conf:.3f}`")
            st.write("Top predictions:")
            idxs = np.argsort(probs)[::-1][:5]
            for i in idxs:
                nm = labels[i] if labels and i < len(labels) else f"class_{i}"
                st.write(f"- {nm}: {probs[i]:.4f}")
        except Exception as e:
            st.error("Inference failed: " + str(e))
    else:
        st.warning("No TFLite model found or interpreter missing. Place edge_impulse_model.tflite in the repo and include tensorflow or tflite-runtime in requirements.")

    # spectrogram preview
    try:
        data, sr = read_audio_bytes(audio_bytes)
        if sr != TARGET_SR:
            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        S = librosa.feature.melspectrogram(y=data.astype(float), sr=sr, n_mels=40)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(S_db, aspect='auto', origin='lower')
        ax.axis('off')
        st.pyplot(fig)
    except Exception:
        pass

st.markdown("---")
st.markdown("<div style='text-align:center; color:#073b46'>Prototype only — not a medical diagnosis tool</div>", unsafe_allow_html=True)
