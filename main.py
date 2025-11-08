# main.py
"""
Vaani — Streamlit app with inline HTML/CSS/JS frontend + TFLite inference
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

# Prefer soundfile if available, but we will fallback to librosa if not.
try:
    import soundfile as sf
    SOUND_FILE_AVAILABLE = True
except Exception:
    sf = None
    SOUND_FILE_AVAILABLE = False

# Try to import tflite interpreter from tflite_runtime, otherwise from tensorflow.lite
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

# Optional Vosk (not required)
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

# Inline frontend HTML/CSS/JS (modern yellow-themed recorder)
FRONTEND_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Vaani — Recorder</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root{
      --bg1: #fff17a;
      --bg2: #ffefb3;
      --accent: #072b34;
      --muted: #0b2f35;
      --card:#ffffff;
      --glass: rgba(255,255,255,0.92);
    }
    html,body{height:100%;margin:0;font-family:"Inter",system-ui,Segoe UI,Roboto,Arial;color:var(--accent);background:linear-gradient(180deg,var(--bg1) 0%, var(--bg2) 100%);}
    .container{max-width:1100px;margin:28px auto;padding:20px;}
    .topbar{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
    .brand{display:flex;align-items:center;gap:12px}
    .logo{
      width:52px;height:52px;border-radius:10px;background:linear-gradient(135deg,#073b46,#0b8a8f);
      display:flex;align-items:center;justify-content:center;color:white;font-weight:800;font-size:20px;
      box-shadow:0 8px 24px rgba(3,27,34,0.12);
    }
    .brand h1{margin:0;font-size:20px;letter-spacing:-0.4px}
    .brand p{margin:0;font-size:12px;color:rgba(11,35,40,0.75)}

    .hero{text-align:center;padding:28px 10px 6px 10px}
    .hero h2{font-size:36px;margin:6px 0 4px 0;font-weight:800}
    .hero p{margin:0;color:rgba(11,35,40,0.85);opacity:0.95;font-size:15px}

    .grid{display:grid;grid-template-columns:1fr;gap:16px;margin-top:18px}
    @media(min-width:880px){ .grid{grid-template-columns: 1fr 420px;} }

    /* Recorder card */
    .card{background:var(--glass);border-radius:16px;padding:18px;box-shadow:0 18px 40px rgba(3,27,34,0.06);border:1px solid rgba(3,27,34,0.04)}
    .card h3{margin:0 0 8px 0}
    .muted{color:rgba(11,35,40,0.7);font-size:14px}

    .rec-row{display:flex;align-items:center;gap:16px}
    @media(max-width:880px){ .rec-row{flex-direction:column;align-items:stretch} }

    .rec-btn{
      width:84px;height:84px;border-radius:50%;border:none;background:linear-gradient(180deg,#ff4f4f,#ff1f1f);cursor:pointer;
      box-shadow:0 12px 28px rgba(255,79,79,0.24);display:flex;align-items:center;justify-content:center;font-size:18px;color:white;font-weight:800;
      transition:transform .12s ease;
    }
    .rec-btn.recording{animation:recPulse 1.2s infinite; transform:scale(1.03);}
    @keyframes recPulse{0%{box-shadow:0 12px 28px rgba(255,79,79,0.24)}50%{box-shadow:0 20px 40px rgba(255,79,79,0.36);transform:scale(1.06)}100%{box-shadow:0 12px 28px rgba(255,79,79,0.24)}}

    .wavebox{flex:1;height:86px;border-radius:12px;background:linear-gradient(90deg,rgba(3,27,34,0.03),rgba(3,27,34,0.02));display:flex;align-items:center;padding:8px 12px;border:1px dashed rgba(3,27,34,0.06)}
    .wave-inner{width:100%;display:flex;gap:4px;align-items:flex-end;justify-content:flex-start;height:68px}

    .wave-bar{width:6px;background:linear-gradient(180deg,#073b46,#0b1220);border-radius:3px;opacity:0.9}

    .info{min-width:140px;display:flex;flex-direction:column;gap:10px;align-items:flex-end}
    .timer{font-weight:800;font-size:18px}
    .controls{display:flex;gap:8px}
    .btn{padding:8px 12px;border-radius:10px;border:none;background:var(--accent);color:white;font-weight:700;cursor:pointer}

    .suggestions{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
    .chip{padding:8px 12px;border-radius:999px;background:rgba(7,59,70,0.06);cursor:pointer;border:1px solid rgba(7,59,70,0.03);font-weight:600}

    .b64area{margin-top:12px}
    textarea{width:100%;height:120px;border-radius:10px;padding:10px;font-family:monospace;font-size:13px;border:1px solid rgba(3,27,34,0.06)}

    /* right column */
    .panel{display:flex;flex-direction:column;gap:12px}
    .panel .card{padding:14px}

    .small{font-size:13px;color:rgba(11,35,40,0.75)}

    .download-btn{background:linear-gradient(180deg,#073b46,#0b8a8f);border:none;color:white;padding:10px 14px;border-radius:12px;font-weight:700;cursor:pointer}

    footer{margin-top:18px;text-align:center;color:rgba(7,59,70,0.7);font-size:13px}
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <div class="brand">
        <div class="logo">V</div>
        <div>
          <h1 style="margin:0">Vaani</h1>
          <p style="margin:0;font-size:12px;color:rgba(11,35,40,0.7)">Voice screening prototype</p>
        </div>
      </div>
      <div style="display:flex;gap:12px;align-items:center">
        <div style="font-size:13px;color:rgba(11,35,40,0.7)">Beta</div>
      </div>
    </div>

    <div class="hero">
      <h2>Record a short sentence</h2>
      <p class="muted">Friendly UI — press Record, speak for 3–8 seconds, stop, then paste Base64 into the Vaani app to analyze.</p>
    </div>

    <div class="grid">
      <div class="card">
        <h3>Voice Recorder</h3>
        <div class="muted">Try one of the suggested sentences or speak naturally.</div>

        <!-- recorder -->
        <div style="margin-top:12px" class="rec-row">
          <div style="display:flex;flex-direction:column;align-items:center;gap:8px">
            <button id="recBtn" class="rec-btn" title="Start recording">●</button>
            <div style="font-size:12px;color:rgba(7,59,70,0.6)">Record</div>
          </div>

          <div class="wavebox">
            <div class="wave-inner" id="waveInner"></div>
          </div>

          <div class="info">
            <div class="timer" id="timer">00:00</div>
            <div class="controls">
              <button id="stopBtn" class="btn" disabled>Stop</button>
              <button id="resetBtn" class="btn" style="background:transparent;color:var(--accent);border:1px solid rgba(3,27,34,0.06)">Reset</button>
            </div>
          </div>
        </div>

        <div class="suggestions" id="suggestions" style="margin-top:14px">
          <div class="chip" data-sent="I have noticed a change in my voice recently.">Change in voice</div>
          <div class="chip" data-sent="I feel my speech is slower than before.">Speech slower</div>
          <div class="chip" data-sent="I have trouble pronouncing certain words.">Pronunciation</div>
          <div class="chip" data-sent="My voice is hoarse or breathy today.">Hoarseness</div>
        </div>

        <div class="b64area">
          <label style="font-weight:700;display:block;margin-bottom:8px">Recorded Base64</label>
          <textarea id="b64out" placeholder="Recorded Base64 will appear here after you stop"></textarea>
          <div style="display:flex;gap:8px;margin-top:8px;align-items:center">
            <button id="copyBtn" class="btn">Copy Base64</button>
            <button id="downloadBtn" class="download-btn">Download WAV</button>
          </div>
          <div style="margin-top:10px" class="small">Tip: Copy the Base64 and paste into the Vaani Streamlit app "Paste Base64 audio" box, then press Load.</div>
        </div>

      </div>

      <!-- right panel -->
      <div class="panel">
        <div class="card">
          <h4 style="margin:0 0 8px 0">Quick instructions</h4>
          <ol style="margin:0;padding-left:18px" class="small">
            <li>Click the red Record button and speak a 3–8 second sentence.</li>
            <li>Click Stop when done — a Base64 textarea will appear.</li>
            <li>Copy & paste the Base64 into the Streamlit Vaani app and click Load.</li>
          </ol>
        </div>

        <div class="card">
          <h4 style="margin:0 0 8px 0">Privacy</h4>
          <div class="small">Recordings remain local to your browser until you copy/paste them to the server. Do not upload private medical data without consent.</div>
        </div>

        <div class="card">
          <h4 style="margin:0 0 8px 0">Try this</h4>
          <div class="small">Read clearly and naturally. Avoid loud background noise.</div>
        </div>
      </div>
    </div>

    <footer>Prototype only — not a medical diagnosis tool</footer>
  </div>

  <script>
    // DOM refs
    const recBtn = document.getElementById('recBtn');
    const stopBtn = document.getElementById('stopBtn');
    const resetBtn = document.getElementById('resetBtn');
    const timerEl = document.getElementById('timer');
    const waveInner = document.getElementById('waveInner');
    const b64out = document.getElementById('b64out');
    const copyBtn = document.getElementById('copyBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const chips = document.querySelectorAll('.chip');

    let mediaRecorder = null;
    let chunks = [];
    let startMs = 0;
    let timerInt = null;

    function pad(n){ return (n<10? '0'+n : n); }
    function format(ms){ const s=Math.floor(ms/1000); return pad(Math.floor(s/60)) + ":" + pad(s%60); }
    function makeWave(num=30){
      waveInner.innerHTML = '';
      for(let i=0;i<num;i++){
        const bar = document.createElement('div');
        bar.className = 'wave-bar';
        bar.style.height = (6 + Math.random()*62) + 'px';
        waveInner.appendChild(bar);
      }
    }
    makeWave(32);

    chips.forEach(c => c.addEventListener('click', ()=> {
      const s = c.dataset.sent || c.innerText;
      navigator.clipboard.writeText(s).then(()=> {
        alert('Sample sentence copied to clipboard — paste into your phone/notes and read it while recording.');
      }).catch(()=> {
        prompt('Copy this sentence and read it while recording:', s);
      });
    }));

    recBtn.addEventListener('click', async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        chunks = [];
        mediaRecorder.ondataavailable = e => chunks.push(e.data);
        mediaRecorder.onstop = async () => {
          const recordedBlob = new Blob(chunks, { type: 'audio/wav' });
          const ab = await recordedBlob.arrayBuffer();
          const u8 = new Uint8Array(ab);
          const b64 = btoa(String.fromCharCode(...u8));
          b64out.value = b64;
          downloadBtn.onclick = () => {
            const url = URL.createObjectURL(recordedBlob);
            const a = document.createElement('a');
            a.href = url; a.download = 'vaani_recording.wav'; document.body.appendChild(a); a.click(); a.remove();
            URL.revokeObjectURL(url);
          };
        };
        mediaRecorder.start();
        recBtn.classList.add('recording');
        recBtn.disabled = true;
        stopBtn.disabled = false;
        startMs = Date.now();
        timerInt = setInterval(()=> timerEl.textContent = format(Date.now()-startMs), 200);
        recBtn._interval = setInterval(()=> {
          const bars = waveInner.children;
          for(let i=0;i<bars.length;i++){
            bars[i].style.height = (8 + Math.random()*62) + 'px';
          }
        }, 160);
      } catch (err) {
        alert('Microphone access error: ' + (err.message || err));
      }
    });

    stopBtn.addEventListener('click', () => {
      if(mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
      recBtn.classList.remove('recording');
      recBtn.disabled = false;
      stopBtn.disabled = true;
      clearInterval(timerInt);
      timerEl.textContent = '00:00';
      if(recBtn._interval) clearInterval(recBtn._interval);
    });

    resetBtn.addEventListener('click', () => {
      if(mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
      b64out.value = '';
      timerEl.textContent = '00:00';
      makeWave(32);
      recBtn.classList.remove('recording');
      recBtn.disabled = false;
      stopBtn.disabled = true;
    });

    copyBtn.addEventListener('click', async () => {
      if(!b64out.value) { alert('No Base64 recorded yet'); return; }
      try{
        await navigator.clipboard.writeText(b64out.value);
        alert('Base64 copied to clipboard. Paste it into the Vaani Streamlit app.');
      }catch(e){
        prompt('Copy this Base64:', b64out.value);
      }
    });
  </script>
</body>
</html>
"""

# Render frontend HTML in app
st.components.v1.html(FRONTEND_HTML, height=640)

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

# Sidebar: show model info if present
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
            # don't allocate tensors here to avoid heavy ops until inference
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
# Helpers for audio & inference
# -----------------------
def read_audio_bytes(audio_bytes: bytes):
    """
    Return (mono float32 array, sample_rate). Use soundfile if available, otherwise librosa/audioread.
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
    # raw PCM vector case
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
    # spectrogram-like case
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
    # fallback: flatten/pad
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
            # try softmax if needed
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
        st.warning("No TFLite model found or interpreter missing. Place edge_impulse_model.tflite in the repo and include tensorflow or tflite-runtime in requirements to enable inference.")

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
