"""
VoiceALS - Edge Impulse TFLite Streamlit app (main.py)

Place your TFLite file next to this script as: edge_impulse_model.tflite
Optional: labels.txt (one label per line)

Run:
    pip install -r requirements.txt
    streamlit run main.py
"""
import streamlit as st
import numpy as np
import io, os, base64, json
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

# Try to import tflite interpreter from tflite_runtime, otherwise try tensorflow
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

# Optional: Vosk offline STT (if installed and a model present)
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False
    
EI_MODEL_PATH = "edge_impulse_model.tflite"   # put your exported TFLite here
LABELS_PATH = "labels.txt"
TARGET_SR = 16000       # default sample-rate (Edge Impulse commonly uses 16k)
EXPECTED_SECONDS = 1.0  # default window length; adjust if your EI project differs

st.set_page_config(page_title="Edge Impulse TFLite Demo", layout="centered")
st.title("Edge Impulse TFLite — Streamlit inference demo")

# Sidebar: load model & show details
interpreter = None
input_details = None
output_details = None
labels = None

# Load labels if present
if os.path.exists(LABELS_PATH):
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = [l.strip() for l in f if l.strip()]
            st.sidebar.success(f"Loaded {len(labels)} labels from {LABELS_PATH}")
    except Exception as e:
        st.sidebar.warning(f"Could not read labels.txt: {e}")

if os.path.exists(EI_MODEL_PATH):
    if TFLiteInterpreter is None:
        st.sidebar.error("No TFLite interpreter available. Install tflite-runtime or tensorflow.")
    else:
        try:
            interpreter = TFLiteInterpreter(model_path=EI_MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            st.sidebar.success(f"Loaded TFLite model: {EI_MODEL_PATH} ({TFLITE_SOURCE})")
            st.sidebar.write("Input details:")
            st.sidebar.write(input_details)
            st.sidebar.write("Output details:")
            st.sidebar.write(output_details)
        except Exception as e:
            st.sidebar.error(f"Failed to load TFLite model: {e}")
else:
    st.sidebar.warning(f"Place a TFLite model at: {EI_MODEL_PATH}")

st.markdown(
    """
**Instructions**
- Upload a short audio clip (WAV/MP3) or use the in-browser recorder (copy base64 and paste).
- The app will preprocess audio to match your TFLite input shape where possible and run inference.
- If your model is quantized (int8/uint8), the app will use the model's quantization params automatically.
"""
)

# ---------------------------
# Utilities
# ---------------------------
def read_audio_bytes(audio_bytes: bytes):
    """Return mono float32 array and sample rate."""
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        return data.astype(np.float32), sr
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr

def quantize_input(arr_float: np.ndarray, input_detail: dict):
    """
    Quantize float numpy array to the input dtype using quantization params from input_detail.
    input_detail['quantization'] gives (scale, zero_point).
    """
    dtype = np.dtype(input_detail['dtype'])
    q_params = input_detail.get('quantization', (0.0, 0))
    scale, zero_point = q_params
    if scale == 0 and zero_point == 0 and np.issubdtype(dtype, np.floating):
        return arr_float.astype(dtype)
    # handle integer quantization
    if np.issubdtype(dtype, np.integer):
        # (float / scale) + zero_point  -> round -> cast
        # safe guard: avoid division by zero
        if scale == 0:
            scale = 1e-8
        arr_q = np.round(arr_float / scale + zero_point).astype(dtype)
        return arr_q
    else:
        return arr_float.astype(dtype)

def preprocess_for_model(audio_bytes: bytes, input_details):
    """
    Create an input tensor that matches the TFLite model input shape & dtype.
    Heuristics for common EI audio models:
      - If input is 1D [1, N] or [N]: treat as raw PCM (samples).
      - If input is 4D [1, frames, bins, 1]: produce mel-spectrogram with bins.
    If your model used a very specific DSP pipeline in Edge Impulse, adapt this function accordingly.
    """
    x, sr = read_audio_bytes(audio_bytes)
    # resample
    if sr != TARGET_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    inp = input_details[0]
    shape = list(inp['shape'])  # might be e.g. [1, 49, 40, 1] or [1, 16000]
    dtype = np.dtype(inp['dtype'])

    # Case: raw PCM vector
    if len(shape) == 2 or len(shape) == 1:
        # samples expected = last element
        n_samples = int(shape[-1])
        # pad/trim
        if x.shape[0] < n_samples:
            x_padded = np.pad(x, (0, n_samples - x.shape[0]), mode='constant')
        else:
            x_padded = x[:n_samples]
        # many EI raw models expect int16 or int8 — scale accordingly if dtype int
        if np.issubdtype(dtype, np.integer):
            if dtype == np.int16:
                arr = (x_padded * 32767.0).astype(np.int16)
            elif dtype == np.int8 or dtype == np.int32 or dtype == np.uint8:
                # use quantization parameters if present
                # convert float to int range first by scaling to [-32767,32767] then quantize
                arr_float = x_padded.astype(np.float32)
                arr = quantize_input(arr_float, inp)
            else:
                arr = x_padded.astype(dtype)
        else:
            arr = x_padded.astype(dtype)
        arr = arr.reshape([1, -1])
        return arr

    # Case: 4D spectrogram MFCC-like: [1, frames, bins, 1]
    if len(shape) == 4:
        _, frames, bins, ch = shape
        desired_frames = int(frames)
        desired_bins = int(bins)

        # Compute mel spectrogram (n_mels = bins)
        S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=desired_bins)
        S_db = librosa.power_to_db(S, ref=np.max)
        # S_db shape: (bins, time_frames) -> transpose
        S_t = S_db.T  # (time_frames, bins)
        # pad/trim to desired_frames
        if S_t.shape[0] < desired_frames:
            pad_amt = desired_frames - S_t.shape[0]
            S_t = np.pad(S_t, ((0, pad_amt), (0,0)), mode='constant', constant_values=(S_t.min(),))
        else:
            S_t = S_t[:desired_frames, :]
        # Normalize
        S_norm = (S_t - S_t.mean()) / (S_t.std() + 1e-9)
        out = S_norm.reshape((1, desired_frames, desired_bins, 1))
        # cast/quantize
        if np.issubdtype(dtype, np.integer):
            out = quantize_input(out.astype(np.float32), inp)
        else:
            out = out.astype(dtype)
        return out

    # Fallback: flatten/pad to product of remaining dims
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
    """Set input tensor and run; return outputs as list of numpy arrays"""
    indie = interpreter.get_input_details()
    outie = interpreter.get_output_details()
    # Some models require specific index for input; we set first input
    interpreter.set_tensor(indie[0]['index'], input_tensor)
    interpreter.invoke()
    outputs = []
    for od in outie:
        outputs.append(interpreter.get_tensor(od['index']))
    return outputs

# ---------------------------
# Recorder HTML (copy base64)
# ---------------------------
recorder_html = """
<p>Record in-browser (start / stop). After stopping a textarea with Base64 will be added to the page — copy its contents and paste it into the 'Base64 audio' box in the app.</p>
<button id="rec">Start</button><button id="stop" disabled>Stop</button>
<p id="status">Idle</p>
<script>
let mediaRecorder, audioChunks=[];
document.getElementById('rec').onclick = async () => {
  const s = await navigator.mediaDevices.getUserMedia({audio:true});
  mediaRecorder = new MediaRecorder(s);
  audioChunks=[];
  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = async () => {
    const blob = new Blob(audioChunks, {type:'audio/wav'});
    const ab = await blob.arrayBuffer();
    const u8 = new Uint8Array(ab);
    const b64 = btoa(String.fromCharCode(...u8));
    const ta = document.createElement('textarea');
    ta.value = b64;
    document.body.appendChild(ta);
    alert('Recording created. Copy the textarea content and paste into the app box.');
  };
  mediaRecorder.start();
  document.getElementById('status').innerText = 'Recording...';
  document.getElementById('rec').disabled = true;
  document.getElementById('stop').disabled = false;
};
document.getElementById('stop').onclick = () => {
  mediaRecorder.stop();
  document.getElementById('status').innerText = 'Stopped';
  document.getElementById('rec').disabled = false;
  document.getElementById('stop').disabled = true;
};
</script>
"""

# ---------------------------
# UI: upload or paste base64
# ---------------------------
st.header("Upload audio or use in-browser recorder")
uploaded = st.file_uploader("Upload audio file (wav, mp3, m4a, ogg)", type=["wav","mp3","m4a","ogg"])
if st.button("Open in-browser recorder (copy base64 and paste)"):
    st.components.v1.html(recorder_html, height=240)

b64 = st.text_area("Paste Base64 audio here (if you used in-browser recorder)", height=140)

audio_bytes = None
if uploaded is not None:
    audio_bytes = uploaded.read()
elif b64 and len(b64) > 50:
    try:
        audio_bytes = base64.b64decode(b64.strip())
    except Exception as e:
        st.error("Base64 decode error: " + str(e))

if audio_bytes is None:
    st.info("Upload an audio file or paste base64 from the recorder to run inference.")
else:
    st.success("Audio received")
    st.audio(audio_bytes)

    # Optional: local Vosk transcription (if installed + model)
    if VOSK_AVAILABLE:
        st.markdown("### Optional: local (Vosk) transcription")
        VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-en-us-0.15")
        if os.path.exists(VOSK_MODEL_PATH):
            try:
                vosk_model = VoskModel(VOSK_MODEL_PATH)
                rec = KaldiRecognizer(vosk_model, TARGET_SR)
                data, sr = read_audio_bytes(audio_bytes)
                if data.dtype.kind == 'f':
                    pcm = (data * 32767).astype(np.int16).tobytes()
                else:
                    pcm = data.tobytes()
                rec.AcceptWaveform(pcm)
                res = json.loads(rec.FinalResult())
                st.write(res.get("text", ""))
            except Exception as e:
                st.warning(f"Vosk failed: {e}")
        else:
            st.info("Vosk installed but no model found at models/... (set VOSK_MODEL_PATH env or place model).")

    # Preprocess -> inference
    if interpreter is None:
        st.warning("No TFLite model loaded. Place edge_impulse_model.tflite in the app folder.")
    else:
        try:
            in_details = interpreter.get_input_details()
            inp_tensor = preprocess_for_model(audio_bytes, in_details)
            outputs = run_tflite(interpreter, inp_tensor)
            # assume first output is class probabilities or logits
            out = outputs[0].squeeze()
            # If it's a multi-dim array, flatten to 1D probabilities
            if out.ndim > 1:
                out = out.flatten()
            # Try to softmax if values are not normalized (not always needed)
            try:
                # If outputs are small/negative, probably logits; apply softmax
                if np.any(out < 0) or np.any(out > 1) or not np.isclose(out.sum(), 1.0):
                    exp = np.exp(out - np.max(out))
                    probs = exp / exp.sum()
                else:
                    probs = out.astype(np.float32)
            except Exception:
                probs = out.astype(np.float32)
            # Show top prediction(s)
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
            st.info("If this fails, inspect model input details shown in the sidebar and adapt preprocess_for_model() to match Edge Impulse DSP settings (sample rate, mel bins, frame length/hop, quantization).")

    # Spectrogram preview
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

st.markdown("---")
st.markdown(
    "Notes: If your TFLite model is quantized (int8/uint8) the app uses quantization params from the model. "
    "If your Edge Impulse pipeline used very specific DSP parameters, update preprocess_for_model() to mirror them exactly."
)
