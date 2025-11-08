# main.py
"""
VoiceALS — Edge Impulse TFLite Streamlit app (styled UI + modern recorder bar)
Drop your edge_impulse_model.tflite + optional labels.txt next to this file.

Run:
    pip install -r requirements.txt
    streamlit run main.py
"""
import streamlit as st
import numpy as np
import io, os, base64, json
import librosa
import matplotlib.pyplot as plt

# Prefer soundfile if available, but fallback to librosa/audioread to avoid libsndfile on cloud
try:
    import soundfile as sf
    SOUND_FILE_AVAILABLE = True
except Exception:
    sf = None
    SOUND_FILE_AVAILABLE = False

# TFLite interpreter
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

# --------------------
# Config
# --------------------
EI_MODEL_PATH = "edge_impulse_model.tflite"
LABELS_PATH = "labels.txt"
TARGET_SR = 16000

st.set_page_config(page_title="VoiceALS", layout="centered", initial_sidebar_state="collapsed")

# ---- Stylish CSS (yellow background, hero, layout) ----
PAGE_CSS = """
<style>
/* page background and typography */
html, body, [data-testid="stAppViewContainer"] > .main {
  background: linear-gradient(180deg, #ffe84d 0%, #ffec99 100%);
  color: #0b1220;
  font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
}

/* narrow the main container and center */
section.main > div.block-container { max-width: 900px; }

/* hero */
.hero {
  text-align: center;
  padding: 36px 12px;
  margin-bottom: 6px;
}
.hero h1 {
  font-size: 34px;
  margin: 6px 0 6px 0;
  letter-spacing: -0.5px;
  font-weight: 700;
}
.hero p.lead {
  margin: 0;
  color: #073b46;
  opacity: 0.9;
}

/* card look for recorder */
.recorder-card {
  background: rgba(255,255,255,0.85);
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 6px 30px rgba(11,18,32,0.08);
  border: 1px solid rgba(11,18,32,0.06);
}

/* record bar layout */
.record-bar {
  display:flex;
  align-items:center;
  gap: 14px;
}
.btn-record {
  width:72px;
  height:72px;
  border-radius:50%;
  border: none;
  background: linear-gradient(180deg,#ff3b3b,#ff0000);
  box-shadow: 0 8px 18px rgba(255,59,59,0.22);
  cursor: pointer;
}
.btn-record.recording {
  animation: pulse 1.2s infinite;
  transform: scale(1.03);
}
@keyframes pulse {
  0% { box-shadow: 0 8px 18px rgba(255,59,59,0.22); }
  50% { box-shadow: 0 12px 30px rgba(255,59,59,0.30); transform: scale(1.06); }
  100% { box-shadow: 0 8px 18px rgba(255,59,59,0.22); transform: scale(1.03); }
}

.waveform {
  flex: 1;
  height:72px;
  background: linear-gradient(90deg, rgba(3,27,34,0.04), rgba(3,27,34,0.02));
  border-radius: 8px;
  display:flex;
  align-items:center;
  justify-content:center;
  padding: 6px 12px;
  border: 1px dashed rgba(3,27,34,0.06);
}

/* timer and controls */
.record-info {
  display:flex;
  flex-direction:column;
  gap:6px;
  min-width:140px;
}
.timer {
  font-weight:600;
  color:#0b1220;
}
.controls {
  display:flex;
  gap:8px;
  align-items:center;
}

.small-btn {
  padding:8px 12px;
  border-radius:8px;
  border:none;
  background: #073b46;
  color: white;
  cursor:pointer;
  font-weight:600;
}

/* responsive */
@media (max-width:640px) {
  .record-bar { flex-direction: column; align-items:stretch; gap:12px; }
  .record-info { min-width:auto; flex-direction:row; justify-content:space-between; }
}
</style>
"""

st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ---- Hero section ----
st.markdown(
    """
    <div class="hero">
      <h1>VoiceALS — accessible early voice screening</h1>
      <p class="lead">Record a short sentence. We extract voice features and run a model to flag possible ALS/Parkinsonian patterns.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar: model info (same as before)
st.sidebar.title("Model & Info")
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [l.strip() for l in f if l.strip()]
            st.sidebar.write(f"Labels loaded: {len(labels)}")
    except Exception:
        st.sidebar.write("labels.txt exists but could not be read.")
else:
    st.sidebar.write("No labels.txt found.")

interpreter = None
if os.path.exists(EI_MODEL_PATH):
    if TFLiteInterpreter is None:
        st.sidebar.error("No TFLite interpreter available (install tensorflow or tflite-runtime).")
    else:
        try:
            interpreter = TFLiteInterpreter(model_path=EI_MODEL_PATH)
            interpreter.allocate_tensors()
            st.sidebar.success("TFLite model loaded")
            st.sidebar.write("Input details:")
            try:
                st.sidebar.write(interpreter.get_input_details())
                st.sidebar.write("Output details:")
                st.sidebar.write(interpreter.get_output_details())
            except Exception:
                pass
        except Exception as e:
            st.sidebar.error("Failed to load TFLite model: " + str(e))
else:
    st.sidebar.info("No model file found: place edge_impulse_model.tflite in repo")

# ---- Recorder HTML (styled horizontal bar with waveform) ----
# NOTE: Because Streamlit's simple html component can't directly pass binary to Python,
# this recorder follows a robust, compatible pattern: it creates a textarea containing base64 audio
# after recording; you copy that and paste into the app box below OR click the 'Load from clipboard' helper.
# For a more integrated recorder install `streamlit-audiorecorder` and enable the alternate path.
RECORDER_HTML = """
<div class="recorder-card">
  <div style="font-weight:700; margin-bottom:8px;">Record your voice</div>
  <div style="color:#073b46; margin-bottom:10px;">Read any suggested sentence for 3–6 seconds. Press stop and copy the generated text area content into the app.</div>
  <div class="record-bar">
    <button id="recBtn" class="btn-record" title="Start recording"></button>
    <div class="waveform" id="wave"> <span style="opacity:0.6">Waveform preview</span> </div>
    <div class="record-info">
      <div class="timer" id="timer">00:00</div>
      <div class="controls">
        <button id="stopBtn" class="small-btn" disabled>Stop</button>
        <button id="resetBtn" class="small-btn" style="background:#f3f3f3;color:#073b46">Reset</button>
      </div>
    </div>
  </div>
  <div style="margin-top:10px; font-size:13px; color:#073b46;">
    After you click Stop, a textarea will appear below this recorder with a Base64 string. Copy and paste that string into the app's Base64 box and click "Load and run".
  </div>
</div>

<script>
const recBtn = document.getElementById('recBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const timerEl = document.getElementById('timer');
const waveEl = document.getElementById('wave');

let mediaRecorder;
let audioChunks = [];
let startTime, timerInt;

function formatTime(ms) {
  const s = Math.floor(ms / 1000);
  const mm = String(Math.floor(s/60)).padStart(2,'0');
  const ss = String(s % 60).padStart(2,'0');
  return mm+':'+ss;
}

recBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    audioChunks = [];
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { 'type' : 'audio/wav; codecs=MS_PCM' });
      const arrayBuffer = await blob.arrayBuffer();
      const bytes = new Uint8Array(arrayBuffer);
      const b64 = btoa(String.fromCharCode(...bytes));
      // create textarea with base64
      const ta = document.createElement('textarea');
      ta.id = 'rec_b64';
      ta.style.width = '100%';
      ta.style.height = '120px';
      ta.style.marginTop = '10px';
      ta.placeholder = 'Base64 audio will appear here — copy and paste into the app input';
      ta.value = b64;
      document.querySelector('.recorder-card').appendChild(ta);
      // enable copy-to-clipboard button
      const copyBtn = document.createElement('button');
      copyBtn.textContent = 'Copy to clipboard';
      copyBtn.className = 'small-btn';
      copyBtn.style.marginTop = '8px';
      copyBtn.onclick = async () => { await navigator.clipboard.writeText(ta.value); alert('Copied to clipboard — now paste into the app'); };
      document.querySelector('.recorder-card').appendChild(copyBtn);
    };
    mediaRecorder.start();
    recBtn.classList.add('recording');
    recBtn.title = 'Recording...';
    stopBtn.disabled = false;
    startTime = Date.now();
    timerInt = setInterval(()=> {
      timerEl.textContent = formatTime(Date.now() - startTime);
    }, 200);
    // simple waveform animation
    waveEl.innerHTML = '';
    for(let i=0;i<40;i++){
      let bar = document.createElement('div');
      bar.style.display='inline-block';
      bar.style.width='5px';
      bar.style.height = (4 + Math.random()*60) + 'px';
      bar.style.marginRight='3px';
      bar.style.background = 'linear-gradient(180deg,#073b46,#0b1220)';
      bar.style.borderRadius='2px';
      bar.style.opacity = 0.85;
      waveEl.appendChild(bar);
    }
  } catch(err) {
    alert('Microphone access denied or not supported: ' + err);
  }
};

stopBtn.onclick = () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    recBtn.classList.remove('recording');
    recBtn.title = 'Start recording';
    stopBtn.disabled = true;
    clearInterval(timerInt);
  }
};

resetBtn.onclick = () => {
  // remove existing textarea if any
  const existing = document.getElementById('rec_b64');
  if (existing) existing.remove();
  // remove copy buttons
  const copyBtn = document.querySelector('.recorder-card button.small-btn:nth-last-child(1)');
  // reset timer
  timerEl.textContent = '00:00';
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
  }
  stopBtn.disabled = true;
  recBtn.classList.remove('recording');
  // clear waveform
  waveEl.innerHTML = '<span style="opacity:0.6">Waveform preview</span>';
};
</script>
"""

# Show styled recorder
st.components.v1.html(RECORDER_HTML, height=300)

st.markdown("---")

# Backwards-compatible load box (where user pastes base64)
st.subheader("Paste Base64 audio here (after recording) and click Load")
b64 = st.text_area("Base64 audio (paste here)", height=140)
if st.button("Load and run inference"):
    if not b64 or len(b64) < 50:
        st.error("Please paste a Base64 audio string (from recorder) or upload a file instead.")
    else:
        try:
            audio_bytes = base64.b64decode(b64.strip())
            st.success("Audio loaded — playing below")
            st.audio(audio_bytes)
            # proceed with the same processing pipeline below
            # we reuse helper functions defined after this block
            # convert and run inference
            # (we call the helper functions defined further down)
            # store bytes in session to be processed later
            st.session_state["app_audio_bytes"] = audio_bytes
        except Exception as e:
            st.error("Base64 decode failed: " + str(e))

# Provide an alternative: upload file
uploaded = st.file_uploader("Or upload a file (wav/mp3/m4a/ogg)", type=["wav","mp3","m4a","ogg"])
if uploaded is not None:
    audio_bytes = uploaded.read()
    st.session_state["app_audio_bytes"] = audio_bytes
    st.audio(audio_bytes)

# Optional hint for advanced users: streamlit-audiorecorder
st.markdown(
    """
    **Pro tip (optional):** If you want an integrated recorder that returns audio bytes directly to Python,
    install the community component `streamlit-audiorecorder` (pip). Then uncomment the `st_audiorec()` code path in the script.
    """
)

# ------------------------
# Audio helpers (same as earlier)
# ------------------------
def read_audio_bytes(audio_bytes: bytes):
    """
    Return mono float32 numpy array and sample rate.
    Uses soundfile (faster) if installed and working; otherwise falls back to librosa/audioread.
    """
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

# ------------------------
# If audio bytes are stored in session, run inference and show results
# ------------------------
if "app_audio_bytes" in st.session_state:
    audio_bytes = st.session_state["app_audio_bytes"]
    st.markdown("## Results")
    st.audio(audio_bytes)

    if interpreter is None:
        st.warning("No TFLite model loaded. Place edge_impulse_model.tflite in the repo to run inference.")
    else:
        try:
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
            st.error("Inference / preprocessing failed: " + str(e))

    # spectrogram preview
    try:
        data, sr = read_audio_bytes(audio_bytes)
        if sr != TARGET_SR:
            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        S = librosa.feature.melspectrogram(y=data.astype(float), sr=sr, n_mels=40)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(S_db, aspect='auto', origin='lower')
        ax.axis('off')
        st.pyplot(fig)
    except Exception:
        pass

# footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#073b46'>Prototype only — not a medical diagnosis tool</div>", unsafe_allow_html=True)
