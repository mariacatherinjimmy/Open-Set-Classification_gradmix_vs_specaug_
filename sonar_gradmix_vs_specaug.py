# ======================================================================================
# PASSIVE SONAR OPEN-SET RECOGNITION  |  GradMix vs SpecAugment ablation  |  KAGGLE SINGLE CELL
# TensorFlow/Keras + librosa. Independent CNN + ONE Conformer-style block per known class.
# Known: cargo, tanker, passengership   |   OOD (evaluation only): tug
#
# Training arms (everything else identical: data split, architecture, optimiser, early stopping,
# calibration, thresholds and decision rule are exactly your working pipeline):
#   none      no augmentation
#   specaug   your existing augmentation (time roll + 2 freq masks + 1 time mask + noise)
#   randmask  specaug + a mask with the SAME area/shape as a CAM mask but at a RANDOM location
#             (control: tells you whether CAM *guidance* matters, not just "more masking")
#   gradmix   specaug + LayerCAM-guided masking of the regions the model currently relies on most
#             (Informed Mixing / GradMix idea, adapted to one-vs-rest per-class models)
#
# Outputs (all in OUT_DIR, re-runs reuse trained weights; delete OUT_DIR or set FORCE_RETRAIN=True to redo):
#   results_all_runs.csv, results_summary.csv/.md, paired_differences.csv, cam_statistics.csv,
#   results/<arm>_seed<k>/{metrics.json, thresholds.json, calibration.json, classification_report.txt,
#   confusion_matrix.csv, recording_level_test_predictions.csv}, plots/*.png, run_summary.json, ZIP.
# ======================================================================================
import os, re, json, math, random, hashlib, warnings, zipfile, gc, time
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, f1_score, balanced_accuracy_score)

# ------------------------------------ CONFIG ------------------------------------------
class CFG:
    DATA_ROOT = Path("/kaggle/input/datasets/acousticmickey/deepship/Deepship")
    OUT_DIR = Path("/kaggle/working/runs_sonar_gradmix")
    # Feature caches from your earlier run are reused if their file names match (same audio config + files).
    EXTRA_FEATURE_DIRS = [Path("/kaggle/working/runs_sonar_v3/features")]
    KNOWN = ["cargo", "tanker", "passengership"]
    OOD = "tug"
    ALIASES = {
        "cargo": ["cargo", "cargo ship", "cargo vessel", "cargo_ship", "cargo-ship"],
        "tanker": ["tanker", "tanker ship", "tanker vessel", "oil tanker", "tanker_ship"],
        "passengership": ["passengership", "passenger", "passenger ship", "passenger vessel", "passenger_ship", "passenger-ship"],
        "tug": ["tug", "tugboat", "tug boat", "tug vessel", "tugboat ship"]}
    SPLIT_SEED = 42                 # fixed: all arms/seeds use the SAME recording-level split
    SEEDS = [42]                    # training seeds. Use [42, 43, 44] for mean +- std (each extra seed = same runtime again)
    ARMS = ["none", "specaug", "randmask", "gradmix"]
    SR = 16000
    WINDOW_SEC = 6.0
    STRIDE_SEC = 3.0
    MIN_KEEP_SEC = 3.0
    N_MELS = 128
    N_FFT = 1024
    HOP_FFT = 256
    FMIN, FMAX = 20, 8000
    VAL_FRAC, TEST_FRAC = 0.20, 0.20
    BATCH_SIZE = 16
    MAX_EPOCHS = 40
    PATIENCE = 6
    LR = 3e-4
    EMBED_DIM = 128
    N_HEADS = 4
    DROPOUT = 0.20
    RECON_WEIGHT = 0.25
    TARGET_KNOWN_ACCEPT = 0.95
    # ---- GradMix / control hyper-parameters (fixed a priori; NOT tuned on the tug recordings) ----
    CAM_LAYER = 3                   # which conv stage LayerCAM reads: 1 (fine) .. 3 (coarse, most semantic)
    GM_WARMUP = 3                   # epochs without CAM masking (CAMs of an untrained model are meaningless)
    GM_PROB = 0.5                   # probability that a training sample is CAM-masked after warm-up
    GM_AREA = (0.10, 0.30)          # fraction of the T-F plane masked, drawn uniformly per sample
    GM_FILL = "mean"                # "mean" = fill with sample mean | "swap" = fill with another sample's content (label unchanged)
    # ---- analysis ----
    CAM_MAX_SEGMENTS = 300          # test segments per class used for CAM statistics
    N_BOOT = 1000                   # stratified recording-level bootstrap resamples
    RESTORE_FROM = "auto"           # "auto" = look for <OUT_DIR name> under /kaggle/input; or give an explicit path; None = off
    FORCE_RETRAIN = False
    ZIP_MODELS = False              # weights stay in /kaggle/working either way

ARM_SPEC = {
    "none":     dict(base=False, cam=None,      title="No augmentation"),
    "specaug":  dict(base=True,  cam=None,      title="SpecAugment-style (roll + masks + noise)"),
    "randmask": dict(base=True,  cam="shifted", title="SpecAugment + random-location mask (area-matched control)"),
    "gradmix":  dict(base=True,  cam="guided",  title="SpecAugment + GradMix (LayerCAM-guided mask)"),
}
for a in CFG.ARMS:
    if a not in ARM_SPEC: raise ValueError(f"Unknown arm {a}; choose from {list(ARM_SPEC)}")
if CFG.GM_FILL not in ("mean", "swap"): raise ValueError("GM_FILL must be 'mean' or 'swap'")

K = CFG.KNOWN
random.seed(CFG.SPLIT_SEED); np.random.seed(CFG.SPLIT_SEED); keras.utils.set_random_seed(CFG.SPLIT_SEED)
try:
    for gpu in tf.config.list_physical_devices("GPU"): tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass
CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
for sub in ["models", "features", "plots", "results"]: (CFG.OUT_DIR / sub).mkdir(exist_ok=True)
# ---- carry results over from an earlier Kaggle version/session (add its Output as a notebook input) ----
import shutil
def restore_previous_outputs():
    cands = [Path(CFG.RESTORE_FROM)] if CFG.RESTORE_FROM not in (None, "auto") else []
    if CFG.RESTORE_FROM == "auto" and Path("/kaggle/input").exists():
        cands = list(Path("/kaggle/input").glob(f"*/{CFG.OUT_DIR.name}")) + list(Path("/kaggle/input").glob(f"*/*/{CFG.OUT_DIR.name}"))
    for src_dir in cands:
        if src_dir.is_dir() and src_dir.resolve() != CFG.OUT_DIR.resolve():
            print(f"Restoring previous outputs from {src_dir} (models, features, results, partial checkpoints)")
            shutil.copytree(src_dir, CFG.OUT_DIR, dirs_exist_ok=True); return True
    return False
if not any((CFG.OUT_DIR / "models").glob("*")): restore_previous_outputs()
print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | librosa {librosa.__version__}")
print("GPU:", tf.config.list_physical_devices("GPU"))
print("Arms:", CFG.ARMS, "| seeds:", CFG.SEEDS)

# --------------------------------- DATA DISCOVERY -------------------------------------
def norm(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
ALIASES = {c: [norm(a) for a in al] for c, al in CFG.ALIASES.items()}
def label_path(p):
    for folder in reversed(p.parts[:-1]):          # directory names only
        f = norm(folder)
        hits = [c for c, als in ALIASES.items() if any(f == a or f.startswith(a) for a in als)]
        if len(hits) == 1: return hits[0]
    return None
root = CFG.DATA_ROOT if CFG.DATA_ROOT.exists() else Path("/kaggle/input")
if not CFG.DATA_ROOT.exists(): print(f"Configured DATA_ROOT not found; searching under {root}")
all_wavs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
if not all_wavs: raise FileNotFoundError(f"No WAV files found under {root}. Check CFG.DATA_ROOT.")
by_class = {c: [] for c in K + [CFG.OOD]}; unknown_files = []
for p in all_wavs:
    lab = label_path(p)
    if lab in by_class: by_class[lab].append(p)
    else: unknown_files.append(p)
print(f"Found {len(all_wavs)} WAVs")
for c in K + [CFG.OOD]: print(f"  {c:15s}: {len(by_class[c])} WAVs")
if unknown_files: print("Unclassified WAVs:", len(unknown_files))
missing = [c for c in K if len(by_class[c]) < 3]
if missing: raise RuntimeError(f"Need at least 3 WAVs per known class for train/val/test: {missing}")
if len(by_class[CFG.OOD]) == 0: raise RuntimeError(f"No OOD '{CFG.OOD}' WAVs found. Fix CFG.ALIASES or DATA_ROOT.")

def split_wavs(paths, val_frac, test_frac, seed):      # recording-level split BEFORE segmentation
    paths = np.asarray([str(p) for p in sorted(paths)], dtype=object)
    n = len(paths); n_test = max(1, int(round(n * test_frac))); n_val = max(1, int(round(n * val_frac)))
    if n_test + n_val >= n: n_test, n_val = 1, 1
    rem, test = train_test_split(paths, test_size=n_test, random_state=seed, shuffle=True)
    train, val = train_test_split(rem, test_size=n_val, random_state=seed + 1, shuffle=True)
    return {"train": [Path(x) for x in train], "val": [Path(x) for x in val], "test": [Path(x) for x in test]}
splits = {c: split_wavs(by_class[c], CFG.VAL_FRAC, CFG.TEST_FRAC, CFG.SPLIT_SEED + i * 17) for i, c in enumerate(K)}
for c in K: print(f"{c:15s} train/val/test WAVs = {len(splits[c]['train'])}/{len(splits[c]['val'])}/{len(splits[c]['test'])}")
ood_paths = sorted(by_class[CFG.OOD])
manifest = {c: {part: [str(p) for p in vals] for part, vals in ss.items()} for c, ss in splits.items()}
manifest[CFG.OOD] = {"evaluation_only": [str(p) for p in ood_paths]}
(CFG.OUT_DIR / "wav_split_manifest.json").write_text(json.dumps(manifest, indent=2))

# ---------------------------------- LOG-MEL FEATURES ----------------------------------
WIN = int(CFG.WINDOW_SEC * CFG.SR); STRIDE = int(CFG.STRIDE_SEC * CFG.SR); MIN_KEEP = int(CFG.MIN_KEEP_SEC * CFG.SR)
T_MAX = int(math.ceil(WIN / CFG.HOP_FFT)) + 1
INPUT_SHAPE = (CFG.N_MELS, T_MAX, 1)
FEAT_CFG = {k: getattr(CFG, k) for k in ["SR", "WINDOW_SEC", "STRIDE_SEC", "MIN_KEEP_SEC", "N_MELS", "N_FFT", "HOP_FFT", "FMIN", "FMAX"]}
print("Log-Mel input:", INPUT_SHAPE)
def audio_windows(path):
    y, _ = librosa.load(str(path), sr=CFG.SR, mono=True)
    if len(y) < MIN_KEEP: return
    if len(y) <= WIN: starts = [0]
    else:
        starts = list(range(0, len(y) - MIN_KEEP + 1, STRIDE))
        last = len(y) - WIN
        if last > 0 and (not starts or starts[-1] < last): starts.append(last)
        starts = sorted(set(starts))
    for s in starts:
        seg = y[s:s + WIN]
        if len(seg) < MIN_KEEP: continue
        if len(seg) < WIN: seg = np.pad(seg, (0, WIN - len(seg)))
        yield seg.astype(np.float32)
def logmel(seg):
    mel = librosa.feature.melspectrogram(y=seg, sr=CFG.SR, n_fft=CFG.N_FFT, hop_length=CFG.HOP_FFT,
        n_mels=CFG.N_MELS, fmin=CFG.FMIN, fmax=CFG.FMAX, power=2.0)
    db = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
    x = np.clip((db + 80.0) / 80.0, 0.0, 1.0)
    if x.shape[1] < T_MAX: x = np.pad(x, ((0, 0), (0, T_MAX - x.shape[1])), constant_values=0.0)
    return x[:, :T_MAX].astype(np.float32)
def file_signature(paths): return [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
def feature_cache(name, paths):
    sig = hashlib.sha256(json.dumps([FEAT_CFG, file_signature(paths)], sort_keys=True).encode()).hexdigest()[:14]
    fn = f"{name}_{sig}.npz"
    for d in [CFG.OUT_DIR / "features"] + [Path(x) for x in CFG.EXTRA_FEATURE_DIRS]:
        fp = d / fn
        if fp.exists():
            try:
                with np.load(fp) as z: return z["X"], z["idx"]
            except Exception: pass
    xs, ids = [], []
    print(f"Extracting features {name}: {len(paths)} WAVs")
    for i, p in enumerate(paths):
        before = len(xs)
        try:
            for seg in audio_windows(p): xs.append(logmel(seg).astype(np.float16)); ids.append(i)
        except Exception as e: print(f"  SKIP {p.name}: {type(e).__name__}: {e}")
        if len(xs) == before: print(f"  WARNING: no usable segment from {p}")
    X = np.stack(xs) if xs else np.empty((0, CFG.N_MELS, T_MAX), np.float16)
    idx = np.asarray(ids, np.int32)
    np.savez_compressed(CFG.OUT_DIR / "features" / fn, X=X, idx=idx)
    return X, idx

F = {c: {part: feature_cache(f"{c}_{part}", splits[c][part]) for part in ["train", "val", "test"]} for c in K}
for c in K:
    for part in ["train", "val", "test"]:
        X, ix = F[c][part]
        if len(X) == 0: raise RuntimeError(f"No valid {part} segments for {c}; inspect WAVs/paths.")
        print(f"{c:15s} {part:5s}: {len(X)} segments from {len(np.unique(ix))} usable WAVs")
OOD_FEAT = None
def get_ood():
    # Tug audio is first touched only after the first arm's models, calibration and thresholds are frozen;
    # it is never used for training, calibration, thresholds, checkpoint selection or hyper-parameters.
    global OOD_FEAT
    if OOD_FEAT is None: OOD_FEAT = feature_cache("tug_FINAL_OOD_TEST", ood_paths)
    return OOD_FEAT

# ------------------------------------ AUGMENTATION ------------------------------------
def augment_specaug(x, rng):
    """Your original augmentation: random circular time shift, 2 mel-bin masks, 1 time mask, small noise."""
    x = x.copy()
    if rng.random() < .5: x = np.roll(x, int(rng.integers(0, x.shape[1])), axis=1)
    fill = float(x.mean())
    for _ in range(2):
        w = int(rng.integers(0, 12)); f0 = int(rng.integers(0, CFG.N_MELS - w + 1)); x[f0:f0 + w, :] = fill
    w = int(rng.integers(0, min(30, T_MAX - 1) + 1)); t0 = int(rng.integers(0, T_MAX - w + 1)); x[:, t0:t0 + w] = fill
    x += rng.normal(0, .01, x.shape).astype(np.float32)
    return np.clip(x, 0, 1)

def cam_region(cam, rng, shifted):
    """Boolean T-F mask covering the top GM_AREA fraction of a CAM. shifted=True keeps shape/area but moves it randomly."""
    if float(cam.max()) <= 0: return None
    if shifted: cam = np.roll(cam, (int(rng.integers(cam.shape[0])), int(rng.integers(cam.shape[1]))), axis=(0, 1))
    a = float(rng.uniform(CFG.GM_AREA[0], CFG.GM_AREA[1])); k = max(1, int(a * cam.size))
    thr = float(np.partition(cam.ravel(), -k)[-k])
    return (cam >= thr) if thr > 0 else (cam > 0)

def apply_cam_mask(xb, yb, cam_fn, rng, shifted):
    """GradMix step on one batch: LayerCAM of the model's evidence for the TRUE label -> mask most-used T-F regions."""
    B = len(xb); sel = np.where(rng.random(B) < CFG.GM_PROB)[0]
    if len(sel) == 0: return xb, 0, 0.0
    cam = np.asarray(cam_fn(xb[..., None], yb))
    done, area = 0, 0.0
    for i in sel:
        m = cam_region(cam[i], rng, shifted)
        if m is None or not m.any(): continue
        if CFG.GM_FILL == "swap":
            j = int(rng.integers(B)); src = xb[j].copy(); xb[i][m] = src[m]
        else:
            xb[i][m] = float(xb[i].mean())
        done += 1; area += float(m.mean())
    return xb, done, area

# ------------------------------- MODEL: CNN + ONE CONFORMER ---------------------------
def conformer_block(x, d, heads, drop, prefix):
    y = layers.LayerNormalization(epsilon=1e-6, name=prefix + "_ffn1_ln")(x)
    y = layers.Dense(d * 2, activation="swish", name=prefix + "_ffn1_in")(y); y = layers.Dropout(drop)(y)
    y = layers.Dense(d, name=prefix + "_ffn1_out")(y); x = layers.Add()([x, 0.5 * y])
    y = layers.LayerNormalization(epsilon=1e-6, name=prefix + "_mha_ln")(x)
    y = layers.MultiHeadAttention(num_heads=heads, key_dim=d // heads, dropout=drop, name=prefix + "_mha")(y, y)
    x = layers.Add()([x, layers.Dropout(drop)(y)])
    y = layers.LayerNormalization(epsilon=1e-6, name=prefix + "_conv_ln")(x)
    y = layers.Conv1D(d * 2, 1, padding="same", name=prefix + "_pw1")(y); y = layers.Activation("swish")(y)
    y = layers.Conv1D(d * 2, 7, padding="same", groups=d * 2, name=prefix + "_dw")(y)
    y = layers.BatchNormalization()(y); y = layers.Activation("swish")(y)
    y = layers.Conv1D(d, 1, padding="same", name=prefix + "_pw2")(y); x = layers.Add()([x, layers.Dropout(drop)(y)])
    y = layers.LayerNormalization(epsilon=1e-6, name=prefix + "_ffn2_ln")(x)
    y = layers.Dense(d * 2, activation="swish")(y); y = layers.Dropout(drop)(y); y = layers.Dense(d)(y)
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, 0.5 * y]))

def build_model(cls, compile_it=True):
    inp = keras.Input(INPUT_SHAPE, name=cls + "_logmel"); x = inp; acts = {}
    for i, f in enumerate([32, 64, 128], 1):
        x = layers.Conv2D(f, 3, strides=(2, 2), padding="same", use_bias=False, name=f"{cls}_cnn{i}")(x)
        x = layers.BatchNormalization(name=f"{cls}_bn{i}")(x)
        x = layers.Activation("swish", name=f"{cls}_act{i}")(x); acts[i] = x
    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1), name=cls + "_freq_mean")(x)
    x = layers.Dense(CFG.EMBED_DIM, name=cls + "_projection")(x)
    x = conformer_block(x, CFG.EMBED_DIM, CFG.N_HEADS, CFG.DROPOUT, cls + "_conf1")   # exactly ONE block
    x = layers.LayerNormalization(epsilon=1e-6, name=cls + "_emb_ln")(x)
    emb = layers.GlobalAveragePooling1D(name=cls + "_embedding")(x)
    logit = layers.Dense(1, name=cls + "_binary_logit")(layers.Dropout(.25)(emb))
    ct = int(math.ceil(T_MAX / 8)); d = layers.Dense(16 * ct * 24, activation="swish", name=cls + "_dec_dense")(emb)
    d = layers.Reshape((16, ct, 24))(d)
    for i, f in enumerate([48, 24, 12], 1):
        d = layers.UpSampling2D((2, 2))(d); d = layers.Conv2D(f, 3, padding="same", activation="swish", name=f"{cls}_dec{i}")(d)
    d = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name=cls + "_recon_out")(d)
    if int(d.shape[2]) > T_MAX: d = layers.Cropping2D(((0, 0), (0, int(d.shape[2]) - T_MAX)))(d)
    elif int(d.shape[2]) < T_MAX: d = layers.ZeroPadding2D(((0, 0), (0, T_MAX - int(d.shape[2]))))(d)
    model = keras.Model(inp, {"recon": d, "cls": logit}, name=cls + "_cnn_one_conformer")
    scorer = keras.Model(inp, [emb, logit], name=cls + "_scorer")
    camer = keras.Model(inp, [acts[CFG.CAM_LAYER], logit], name=cls + "_camer")   # shares all weights
    if compile_it:
        model.compile(optimizer=keras.optimizers.Adam(CFG.LR),
            loss={"recon": "mse", "cls": keras.losses.BinaryCrossentropy(from_logits=True)},
            loss_weights={"recon": CFG.RECON_WEIGHT, "cls": 1.0})
    return model, scorer, camer

def make_cam_fn(camer):
    """LayerCAM (Jiang et al., 2021): sum_c ReLU(dScore/dA_c) * A_c, ReLU, upsample, per-sample max-normalise.
    Score = evidence for the TRUE label of the one-vs-rest head (logit if y=1, -logit if y=0)."""
    @tf.function(input_signature=[tf.TensorSpec((None, CFG.N_MELS, T_MAX, 1), tf.float32), tf.TensorSpec((None, 1), tf.float32)])
    def cam_fn(x, y):
        with tf.GradientTape() as tape:
            A, logit = camer(x, training=False)
            score = tf.reduce_sum((2.0 * y - 1.0) * logit)
        G = tape.gradient(score, A)
        cam = tf.nn.relu(tf.reduce_sum(tf.nn.relu(G) * A, axis=-1, keepdims=True))
        cam = tf.image.resize(cam, (CFG.N_MELS, T_MAX), method="bilinear")[..., 0]
        return cam / (tf.reduce_max(cam, axis=[1, 2], keepdims=True) + 1e-8)
    return lambda x, y: cam_fn(tf.convert_to_tensor(x, tf.float32), tf.convert_to_tensor(y, tf.float32)).numpy()

def score_segments(scorer, X, bs=64):
    zs, ls = [], []
    for i in range(0, len(X), bs):
        z, l = scorer(X[i:i + bs].astype(np.float32)[..., None], training=False)
        zs.append(z.numpy()); ls.append(l.numpy().reshape(-1))
    return (np.concatenate(zs).astype(np.float64), np.concatenate(ls).astype(np.float64)) if zs else (np.empty((0, CFG.EMBED_DIM)), np.empty(0))
def maha(z, mu, prec):
    delta = z - mu[None, :]
    return np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", delta, prec, delta), 0.0))

# ------------------------------------ TRAINING ----------------------------------------
def make_items(pos, negs, rng):
    nneg = max(1, len(pos) // max(1, len(negs)))
    items = [(pos, i, 1.) for i in range(len(pos))]
    for arr in negs.values():
        if len(arr):
            sel = rng.choice(len(arr), size=nneg, replace=len(arr) < nneg)
            items.extend((arr, int(i), 0.) for i in sel)
    return items
def iter_batches(items, order, bs):
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        xb = np.stack([items[int(j)][0][items[int(j)][1]] for j in sel]).astype(np.float32)
        yb = np.asarray([[items[int(j)][2]] for j in sel], np.float32)
        yield xb, yb
def eval_loss(model, items, bs=64):
    tot, n = 0.0, 0
    for xb, yb in iter_batches(items, np.arange(len(items)), bs):
        xt = xb[..., None]
        out = model.test_on_batch(xt, {"recon": xt, "cls": yb}, return_dict=True)
        tot += float(out["loss"]) * len(xb); n += len(xb)
    return tot / max(n, 1)

def model_dir(arm, seed, cls): return CFG.OUT_DIR / "models" / f"{arm}_seed{seed}" / cls
def tag_for(cls, arm, seed):
    data = {c: {part: file_signature(splits[c][part]) for part in ["train", "val"]} for c in K}
    keys = ["EMBED_DIM", "N_HEADS", "DROPOUT", "RECON_WEIGHT", "LR", "BATCH_SIZE", "MAX_EPOCHS", "PATIENCE", "SPLIT_SEED"]
    if ARM_SPEC[arm]["cam"]: keys += ["CAM_LAYER", "GM_WARMUP", "GM_PROB", "GM_AREA", "GM_FILL"]
    conf = {k: getattr(CFG, k) for k in keys}
    return hashlib.sha256(json.dumps([cls, arm, seed, ARM_SPEC[arm], conf, FEAT_CFG, data], sort_keys=True, default=str).encode()).hexdigest()

def get_or_train(cls, arm, seed):
    spec = ARM_SPEC[arm]; ci = K.index(cls)
    cdir = model_dir(arm, seed, cls); cdir.mkdir(parents=True, exist_ok=True)
    wfile, mfile, mufile, pfile = cdir / "best.weights.h5", cdir / "metadata.json", cdir / "mean.npy", cdir / "precision.npy"
    tag = tag_for(cls, arm, seed); reuse = False
    if not CFG.FORCE_RETRAIN and all(p.exists() for p in [wfile, mfile, mufile, pfile]):
        try: reuse = json.loads(mfile.read_text()).get("tag") == tag
        except Exception: reuse = False
    if reuse:
        model, scorer, camer = build_model(cls, compile_it=False); model.load_weights(str(wfile))
        meta = json.loads(mfile.read_text())
        print(f"  [{arm}|seed{seed}|{cls}] loaded cached weights (best val loss {meta.get('best_val_loss', float('nan')):.4f}, {meta.get('epochs_ran')} epochs)")
        return {"model": model, "scorer": scorer, "camer": camer, "mu": np.load(mufile), "prec": np.load(pfile),
                "best_val_loss": float(meta.get("best_val_loss", float("nan")))}
    keras.utils.set_random_seed(seed * 100 + ci)
    model, scorer, camer = build_model(cls, compile_it=True)
    cam_fn = make_cam_fn(camer) if spec["cam"] else None
    rng = np.random.default_rng(seed * 1000 + ci)
    others = [o for o in K if o != cls]
    pos_tr = F[cls]["train"][0]; neg_tr = {o: F[o]["train"][0] for o in others}
    pos_va = F[cls]["val"][0];   neg_va = {o: F[o]["val"][0] for o in others}
    val_items = make_items(pos_va, neg_va, np.random.default_rng(CFG.SPLIT_SEED + 123))   # fixed, un-augmented
    # ---- resume support: best weights + epoch counter are checkpointed EVERY epoch, so an interrupted run continues ----
    pw, pm_f = cdir / "partial.weights.h5", cdir / "partial.json"
    start_ep, best, best_w, bad, log = 1, math.inf, None, 0, []
    if not CFG.FORCE_RETRAIN and pw.exists() and pm_f.exists():
        try:
            pm = json.loads(pm_f.read_text())
            if pm.get("tag") == tag:
                model.load_weights(str(pw)); best, bad, log, start_ep = float(pm["best"]), int(pm["bad"]), pm["log"], int(pm["epoch"]) + 1
                best_w = model.get_weights()
                print(f"  [{arm}|seed{seed}|{cls}] RESUMING after epoch {pm['epoch']} (best val {best:.4f}); optimizer state restarts")
        except Exception as e: print("  could not resume partial checkpoint, restarting this model:", e)
    for ep in range(start_ep, CFG.MAX_EPOCHS + 1):
        t0 = time.time(); items = make_items(pos_tr, neg_tr, rng); order = rng.permutation(len(items))
        use_cam = cam_fn is not None and ep > CFG.GM_WARMUP
        tl, n_seen, n_masked, area_sum = 0.0, 0, 0, 0.0
        for xb, yb in iter_batches(items, order, CFG.BATCH_SIZE):
            if spec["base"]: xb = np.stack([augment_specaug(x, rng) for x in xb])
            if use_cam:
                xb, nm, ar = apply_cam_mask(xb, yb, cam_fn, rng, spec["cam"] == "shifted"); n_masked += nm; area_sum += ar
            xt = xb[..., None]
            out = model.train_on_batch(xt, {"recon": xt, "cls": yb}, return_dict=True)
            tl += float(out["loss"]) * len(xb); n_seen += len(xb)
        tr_loss = tl / max(n_seen, 1); va_loss = eval_loss(model, val_items)
        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "cam_masked_fraction": n_masked / max(n_seen, 1),
                    "cam_mean_masked_area": area_sum / max(n_masked, 1), "seconds": time.time() - t0})
        improved = va_loss < best - 1e-6
        if improved: best, best_w, bad = va_loss, model.get_weights(), 0; model.save_weights(str(pw))
        else: bad += 1
        pm_f.write_text(json.dumps({"tag": tag, "epoch": ep, "best": best, "bad": bad, "log": log}))
        pd.DataFrame(log).to_csv(cdir / "training_log.csv", index=False)
        print(f"  [{arm}|seed{seed}|{cls}] ep{ep:02d} train={tr_loss:.4f} val={va_loss:.4f}{' *' if improved else ''}"
              f"{f' masked={n_masked / max(n_seen, 1):.2f}' if use_cam else ''} ({time.time() - t0:.0f}s)")
        if bad >= CFG.PATIENCE: print(f"  early stop at epoch {ep}"); break
    model.set_weights(best_w); model.save_weights(str(wfile))
    for f_ in (pw, pm_f):
        if f_.exists(): f_.unlink()
    pd.DataFrame(log).to_csv(cdir / "training_log.csv", index=False)
    ztr, _ = score_segments(scorer, pos_tr)                      # class statistics on clean (un-augmented) training positives
    lw = LedoitWolf().fit(ztr); mu, prec = lw.location_, lw.precision_
    np.save(mufile, mu); np.save(pfile, prec)
    mfile.write_text(json.dumps({"class": cls, "arm": arm, "seed": seed, "tag": tag, "epochs_ran": len(log), "best_val_loss": best,
                                 "architecture": "CNN + one Conformer-style block"}, indent=2))
    return {"model": model, "scorer": scorer, "camer": camer, "mu": mu, "prec": prec, "best_val_loss": float(best)}

# --------------------------- RECORDING-LEVEL SCORING / DECISION ------------------------
def score_set(art, X, idx, paths, true_label):
    if len(X) == 0: return pd.DataFrame(columns=["wav_path", "true_label", "n_segments"])
    outs = {}
    for c in K:
        z, l = score_segments(art[c]["scorer"], X); outs[c] = (l, maha(z, art[c]["mu"], art[c]["prec"]))
    rows = []
    for wi in sorted(np.unique(idx)):
        mask = idx == wi; row = {"wav_path": str(paths[int(wi)]), "true_label": true_label, "n_segments": int(mask.sum())}
        for c in K:
            logits, dist = outs[c]; row[f"raw_{c}"] = float(np.mean(logits[mask])); row[f"D_{c}"] = float(np.median(dist[mask]))
        rows.append(row)
    return pd.DataFrame(rows)
def fit_calibration(val_raw):
    CAL = {}
    for c in K:
        x = val_raw[f"raw_{c}"].to_numpy().reshape(-1, 1); y = (val_raw["true_label"].to_numpy() == c).astype(int)
        if len(np.unique(y)) < 2: raise RuntimeError(f"Validation data lacks positive/negative examples for calibration of {c}.")
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs").fit(x, y)
        ds = float(np.median(val_raw.loc[val_raw.true_label == c, f"D_{c}"].to_numpy()))
        CAL[c] = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]), "dist_scale": max(ds, 1e-8)}
    return CAL
def add_cal(df, CAL):
    for c in K:
        df[f"L_{c}"] = CAL[c]["a"] * df[f"raw_{c}"] + CAL[c]["b"]; df[f"R_{c}"] = df[f"D_{c}"] / CAL[c]["dist_scale"]
    return df
def decision_components(df):
    L = df[[f"L_{c}" for c in K]].to_numpy(float); R = df[[f"R_{c}" for c in K]].to_numpy(float)
    win = np.argmax(L, axis=1); rr = np.arange(len(df))
    return np.array(K, dtype=object)[win], L[rr, win], R[rr, win]
def apply_decision(df, tau_l, tau_r):
    winner, lmax, rwin = decision_components(df)
    pred = winner.copy(); pred[(lmax < tau_l) | (rwin > tau_r)] = "OOD"
    return pred, lmax, rwin
def wilson(k, n, z=1.96):
    if n == 0: return [float("nan"), float("nan")]
    p = k / n; den = 1 + z * z / n; center = (p + z * z / (2 * n)) / den; half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]
def strat_bootstrap(true_label, Lmax, Rwin, known_mask, closed_ok, n_boot, seed=0):
    rng = np.random.default_rng(seed); groups = [np.where(true_label == g)[0] for g in np.unique(true_label)]
    vals = {"auroc_Lmax": [], "auroc_negR": [], "closed_acc": []}
    for _ in range(n_boot):
        ii = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]); km = known_mask[ii]
        if km.all() or (~km).all(): continue
        vals["auroc_Lmax"].append(roc_auc_score(km.astype(int), Lmax[ii])); vals["auroc_negR"].append(roc_auc_score(km.astype(int), -Rwin[ii]))
        vals["closed_acc"].append(float(closed_ok[ii][km].mean()))
    return {k: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else [float("nan")] * 2) for k, v in vals.items()}

def evaluate(art, arm, seed, first_report):
    rd = CFG.OUT_DIR / "results" / f"{arm}_seed{seed}"; rd.mkdir(parents=True, exist_ok=True)
    val_raw = pd.concat([score_set(art, F[c]["val"][0], F[c]["val"][1], splits[c]["val"], c) for c in K], ignore_index=True)
    if val_raw.empty: raise RuntimeError("No usable known validation recordings; cannot calibrate thresholds.")
    CAL = fit_calibration(val_raw); val_raw = add_cal(val_raw, CAL)
    _, vl, vr = decision_components(val_raw)                       # thresholds from the SAME winner scores used at inference
    tau_L = float(np.quantile(vl, 1.0 - CFG.TARGET_KNOWN_ACCEPT, method="higher")); tau_R = float(np.quantile(vr, CFG.TARGET_KNOWN_ACCEPT, method="higher"))
    vp, _, _ = apply_decision(val_raw, tau_L, tau_R)
    val_accept = float(np.mean(vp != "OOD")); val_correct = float(np.mean(vp == val_raw.true_label.to_numpy()))
    thresholds = {"tau_L": tau_L, "tau_R": tau_R, "n_validation_recordings": int(len(val_raw)), "target_marginal_gate_acceptance": CFG.TARGET_KNOWN_ACCEPT,
                  "actual_validation_combined_acceptance": val_accept, "validation_accuracy_including_rejection": val_correct}
    known_test = add_cal(pd.concat([score_set(art, F[c]["test"][0], F[c]["test"][1], splits[c]["test"], c) for c in K], ignore_index=True), CAL)
    oX, oidx = get_ood(); ood_df = score_set(art, oX, oidx, ood_paths, "OOD")
    if not ood_df.empty: ood_df = add_cal(ood_df, CAL)
    else: print("WARNING: no usable OOD WAVs; OOD metrics cannot be calculated.")
    test = pd.concat([known_test, ood_df], ignore_index=True)
    pred, Lmax, Rwin = apply_decision(test, tau_L, tau_R)
    test["predicted_label"] = pred; test["L_max"] = Lmax; test["R_winner"] = Rwin
    labels = K + ["OOD"]; yt = test.true_label.to_numpy(dtype=object); yp = pred.astype(object)
    cm = confusion_matrix(yt, yp, labels=labels)
    known_mask = yt != "OOD"; ood_mask = ~known_mask
    closed_pred = np.array(K, dtype=object)[test[[f"L_{c}" for c in K]].to_numpy(float).argmax(1)]; closed_ok = closed_pred == yt
    ood_n = int(ood_mask.sum()); ood_rej_n = int(np.sum(yp[ood_mask] == "OOD")); ci = wilson(ood_rej_n, ood_n)
    m = {"arm": arm, "seed": seed,
         "closed_set_accuracy_known_only": float(np.mean(closed_ok[known_mask])),
         "known_acceptance_rate": float(np.mean(yp[known_mask] != "OOD")),
         "known_accuracy_including_rejection": float(np.mean(yp[known_mask] == yt[known_mask])),
         "ood_rejection_rate": ood_rej_n / ood_n if ood_n else float("nan"), "ood_rejection_count": ood_rej_n, "n_ood_recordings": ood_n,
         "ood_rejection_wilson95_lo": ci[0], "ood_rejection_wilson95_hi": ci[1],
         "macro_f1_known_plus_ood_rejection": float(f1_score(yt, yp, labels=labels, average="macro", zero_division=0)),
         "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
         "n_known_test_recordings": int(known_mask.sum()), "validation_combined_acceptance_rate": val_accept,
         "validation_accuracy_including_rejection": val_correct, "tau_L": tau_L, "tau_R": tau_R,
         "mean_best_val_loss": float(np.mean([art[c]["best_val_loss"] for c in K]))}
    for i, c in enumerate(labels):
        n = int(cm[i].sum()); m[f"recall_{c}"] = float(cm[i, i] / n) if n else float("nan")
    if known_mask.any() and ood_mask.any():
        km = known_mask.astype(int)
        m["auroc_known_vs_ood_Lmax"] = float(roc_auc_score(km, Lmax)); m["auroc_known_vs_ood_negative_distance"] = float(roc_auc_score(km, -Rwin))
        m["aupr_ood_using_negative_Lmax"] = float(average_precision_score((~known_mask).astype(int), -Lmax))
        fpr, tpr, _ = roc_curve(km, Lmax); valid = np.where(tpr >= .95)[0]
        m["score_only_fpr_at_95_known_tpr"] = float(fpr[valid[0]]) if len(valid) else float("nan")
        m["combined_rule_ood_false_acceptance_rate"] = float(np.mean(yp[ood_mask] != "OOD"))
        bs = strat_bootstrap(yt, Lmax, Rwin, known_mask, closed_ok, CFG.N_BOOT)
        for k, (lo, hi) in bs.items(): m[f"boot95_{k}_lo"] = lo; m[f"boot95_{k}_hi"] = hi
    report = classification_report(yt, yp, labels=labels, zero_division=0, digits=4)
    test.to_csv(rd / "recording_level_test_predictions.csv", index=False)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(rd / "confusion_matrix.csv")
    (rd / "metrics.json").write_text(json.dumps(m, indent=2)); (rd / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (rd / "calibration.json").write_text(json.dumps(CAL, indent=2)); (rd / "classification_report.txt").write_text(report)
    print(f"\n--- {arm} | seed {seed} ---  known-only acc {m['closed_set_accuracy_known_only']:.3f} | known acc incl. rejection "
          f"{m['known_accuracy_including_rejection']:.3f} | OOD rejected {ood_rej_n}/{ood_n} | macro-F1 {m['macro_f1_known_plus_ood_rejection']:.3f} | "
          f"AUROC(Lmax) {m.get('auroc_known_vs_ood_Lmax', float('nan')):.3f} | AUROC(-R) {m.get('auroc_known_vs_ood_negative_distance', float('nan')):.3f}")
    if first_report:
        print(report); print("Confusion matrix (rows=true, cols=pred):\n", pd.DataFrame(cm, index=labels, columns=labels))
    return {"metrics": m, "test": test, "cm": cm, "CAL": CAL, "thresholds": thresholds, "labels": labels}

# <<MAIN>>
# ----------------------------------- RUN EXPERIMENTS ----------------------------------
VIZ_SEED = CFG.SEEDS[0]; VIZ, CALS, TESTS = {}, {}, {}; RUNS = []
t_start = time.time()
for seed in CFG.SEEDS:
    for arm in CFG.ARMS:
        print("\n" + "=" * 92 + f"\nARM: {arm}  ({ARM_SPEC[arm]['title']})  |  seed {seed}\n" + "=" * 92)
        art = {cls: get_or_train(cls, arm, seed) for cls in K}
        res = evaluate(art, arm, seed, first_report=(seed == VIZ_SEED))
        RUNS.append(res["metrics"]); TESTS[(arm, seed)] = res
        if seed == VIZ_SEED: VIZ[arm] = art; CALS[arm] = res["CAL"]
        else: del art
        gc.collect()
ALL = pd.DataFrame(RUNS); ALL.to_csv(CFG.OUT_DIR / "results_all_runs.csv", index=False)
print(f"\nAll arms finished in {(time.time() - t_start) / 60:.1f} min. Raw per-run table -> results_all_runs.csv")

# ------------------------------- SUMMARY + PAIRED DIFFERENCES --------------------------
MET = ["closed_set_accuracy_known_only", "known_accuracy_including_rejection", "ood_rejection_rate", "macro_f1_known_plus_ood_rejection",
       "balanced_accuracy", "auroc_known_vs_ood_Lmax", "auroc_known_vs_ood_negative_distance", "aupr_ood_using_negative_Lmax",
       "score_only_fpr_at_95_known_tpr", "combined_rule_ood_false_acceptance_rate"] + [f"recall_{c}" for c in K]
MET = [m for m in MET if m in ALL.columns]
num_rows, txt_rows = [], []
for arm in CFG.ARMS:
    sub = ALL[ALL.arm == arm]; nr = {"arm": arm, "n_seeds": len(sub)}; tr = {"arm": arm, "n_seeds": len(sub)}
    for mname in MET:
        v = sub[mname].astype(float).to_numpy(); mu_, sd_ = float(np.nanmean(v)), (float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"))
        nr[mname + "_mean"] = mu_; nr[mname + "_std"] = sd_; tr[mname] = f"{mu_:.3f}" + (f" ± {sd_:.3f}" if len(v) > 1 else "")
    num_rows.append(nr); txt_rows.append(tr)
SUMMARY_NUM, SUMMARY = pd.DataFrame(num_rows), pd.DataFrame(txt_rows)
SUMMARY_NUM.to_csv(CFG.OUT_DIR / "results_summary_numeric.csv", index=False); SUMMARY.to_csv(CFG.OUT_DIR / "results_summary.csv", index=False)
md = ["| " + " | ".join(SUMMARY.columns) + " |", "|" + "---|" * len(SUMMARY.columns)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in SUMMARY.to_numpy()]
(CFG.OUT_DIR / "results_summary.md").write_text("\n".join(md))
print("\n=== RECORDING-LEVEL RESULTS BY ARM (mean ± std over seeds; std needs >=2 seeds) ===")
print(SUMMARY.T.to_string(header=False))
pairs = [("gradmix", "specaug"), ("gradmix", "randmask"), ("specaug", "none")]; prow = []
for a, b in pairs:
    if a in CFG.ARMS and b in CFG.ARMS:
        for mname in MET:
            P = ALL.pivot(index="seed", columns="arm", values=mname); d = (P[a] - P[b]).dropna().to_numpy(float)
            if len(d): prow.append({"comparison": f"{a} - {b}", "metric": mname, "mean_diff": float(d.mean()),
                                    "std_diff": float(d.std(ddof=1)) if len(d) > 1 else float("nan"), "n_seeds": len(d)})
PAIRED = pd.DataFrame(prow); PAIRED.to_csv(CFG.OUT_DIR / "paired_differences.csv", index=False)
if len(PAIRED):
    print("\n=== PAIRED DIFFERENCES (same seed/split; positive = first arm better, except FPR/false-acceptance where lower is better) ===")
    print(PAIRED[PAIRED.metric.isin(["closed_set_accuracy_known_only", "known_accuracy_including_rejection", "ood_rejection_rate",
          "auroc_known_vs_ood_Lmax", "auroc_known_vs_ood_negative_distance", "macro_f1_known_plus_ood_rejection"])].to_string(index=False))

# ---------------------------------- CAM ANALYSIS (viz seed) ----------------------------
CAMFN = {}
def get_camfn(arm, cls):
    if (arm, cls) not in CAMFN: CAMFN[(arm, cls)] = make_cam_fn(VIZ[arm][cls]["camer"])
    return CAMFN[(arm, cls)]
def cam_for(arm, cls, X, bs=32):
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].astype(np.float32)[..., None]; out.append(get_camfn(arm, cls)(xb, np.ones((len(xb), 1), np.float32)))
    return np.concatenate(out) if out else np.empty((0, CFG.N_MELS, T_MAX), np.float32)
def cam_stats(cam):
    ok = cam.reshape(len(cam), -1).sum(1) > 0; cam = cam[ok]
    if len(cam) == 0: return {"entropy": float("nan"), "area_ge_0.5": float("nan"), "freq_centroid_bin": float("nan"), "freq_bandwidth_bins": float("nan")}, np.full(CFG.N_MELS, np.nan)
    flat = cam.reshape(len(cam), -1); p = flat / (flat.sum(1, keepdims=True) + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum(1) / math.log(flat.shape[1]); pf = p.reshape(cam.shape).sum(2); fb = np.arange(CFG.N_MELS)
    cen = (pf * fb).sum(1); bw = np.sqrt((pf * (fb[None] - cen[:, None]) ** 2).sum(1))
    return {"entropy": float(H.mean()), "area_ge_0.5": float((cam >= .5).mean()), "freq_centroid_bin": float(cen.mean()), "freq_bandwidth_bins": float(bw.mean())}, pf.mean(0)
rngv = np.random.default_rng(0); CAMROWS, PROFILES = [], {}
oX, oidx = get_ood()
for arm in CFG.ARMS:
    for c in K:                                                 # true-class evidence from the true class's own model
        Xc = F[c]["test"][0]; sel = rngv.choice(len(Xc), size=min(CFG.CAM_MAX_SEGMENTS, len(Xc)), replace=False)
        st, prof = cam_stats(cam_for(arm, c, Xc[sel])); CAMROWS.append({"arm": arm, "segments_from": c, "model": c, "n_segments": len(sel), **st}); PROFILES[(arm, c)] = prof
    if len(oX):                                                 # tug: evidence of the model that wins on each segment
        sel = rngv.choice(len(oX), size=min(CFG.CAM_MAX_SEGMENTS, len(oX)), replace=False); Xs = oX[sel]
        lg = {c: score_segments(VIZ[arm][c]["scorer"], Xs)[1] for c in K}
        Lc = np.stack([CALS[arm][c]["a"] * lg[c] + CALS[arm][c]["b"] for c in K], 1); win = Lc.argmax(1); cams = np.zeros((len(Xs), CFG.N_MELS, T_MAX), np.float32)
        for ki, c in enumerate(K):
            if (win == ki).any(): cams[win == ki] = cam_for(arm, c, Xs[win == ki])
        st, prof = cam_stats(cams); CAMROWS.append({"arm": arm, "segments_from": CFG.OOD, "model": "winner", "n_segments": len(Xs), **st}); PROFILES[(arm, CFG.OOD)] = prof
CAMDF = pd.DataFrame(CAMROWS); CAMDF.to_csv(CFG.OUT_DIR / "cam_statistics.csv", index=False)
print("\n=== CAM ATTRIBUTION STATISTICS (higher entropy / bandwidth / area = attribution spread over more of the T-F plane) ===")
print(CAMDF.groupby("arm")[["entropy", "area_ge_0.5", "freq_bandwidth_bins"]].mean().reindex(CFG.ARMS).round(4).to_string())

# -------------------------------------- PLOTS ----------------------------------------
def savefig(name):
    plt.tight_layout(); plt.savefig(CFG.OUT_DIR / "plots" / name, dpi=150, bbox_inches="tight"); plt.show(); plt.close()
ARMC = {"none": "#8c8c8c", "specaug": "#1f77b4", "randmask": "#ff7f0e", "gradmix": "#2ca02c"}
# 1) metrics by arm
pm = [("closed_set_accuracy_known_only", "Closed-set acc (known only)"), ("known_accuracy_including_rejection", "Known acc incl. rejection"),
      ("ood_rejection_rate", "Tug rejection rate"), ("macro_f1_known_plus_ood_rejection", "Macro-F1 (4 labels)"),
      ("auroc_known_vs_ood_Lmax", "AUROC known-vs-OOD (Lmax)"), ("auroc_known_vs_ood_negative_distance", "AUROC known-vs-OOD (-distance)")]
pm = [(k, t) for k, t in pm if k in ALL.columns]
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5)); axs = axs.ravel()
for ax, (k, t) in zip(axs, pm):
    for i, arm in enumerate(CFG.ARMS):
        v = ALL.loc[ALL.arm == arm, k].astype(float).to_numpy(); sd = np.std(v, ddof=1) if len(v) > 1 else 0
        ax.bar(i, np.mean(v), yerr=sd, color=ARMC[arm], alpha=.8, capsize=4); ax.scatter(np.full(len(v), i), v, c="k", s=14, zorder=3)
    ax.set_xticks(range(len(CFG.ARMS))); ax.set_xticklabels(CFG.ARMS, rotation=20); ax.set_title(t, fontsize=10); ax.set_ylim(0, 1.05)
fig.suptitle(f"Augmentation ablation (mean ± std over {len(CFG.SEEDS)} seed(s); dots = individual seeds)"); savefig("fig1_metrics_by_arm.png")
# 2) CAM overlays on real test segments
EX = []
for c in K:
    X, idx = F[c]["test"]
    if len(X): ii = np.where(idx == np.unique(idx)[0])[0]; EX.append((c, X[ii[len(ii) // 2]].astype(np.float32)))
if len(oX): ii = np.where(oidx == np.unique(oidx)[0])[0]; EX.append((CFG.OOD, oX[ii[len(ii) // 2]].astype(np.float32)))
ext = [0, CFG.WINDOW_SEC, 0, CFG.N_MELS]
fig, axs = plt.subplots(len(CFG.ARMS), len(EX), figsize=(4.3 * len(EX), 2.9 * len(CFG.ARMS)), squeeze=False)
for r, arm in enumerate(CFG.ARMS):
    for ci_, (lab, x) in enumerate(EX):
        if lab in K: mc = lab
        else:
            lg = {c: float(score_segments(VIZ[arm][c]["scorer"], x[None])[1][0]) for c in K}
            mc = max(K, key=lambda c: CALS[arm][c]["a"] * lg[c] + CALS[arm][c]["b"])
        cam = cam_for(arm, mc, x[None])[0]; ax = axs[r, ci_]
        ax.imshow(x, origin="lower", aspect="auto", cmap="gray", vmin=0, vmax=1, extent=ext)
        ax.imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=.45, vmin=0, vmax=1, extent=ext)
        ax.set_title(f"{lab} segment | model: {mc}" if r == 0 else f"model: {mc}", fontsize=8)
        if ci_ == 0: ax.set_ylabel(f"{arm}\nmel bin", fontsize=9)
        if r == len(CFG.ARMS) - 1: ax.set_xlabel("time (s)")
fig.suptitle("LayerCAM overlays on test log-Mel segments (red = regions the model's evidence relies on)"); savefig("fig2_cam_overlays.png")
# 3) frequency attribution profiles
labs = K + ([CFG.OOD] if len(oX) else [])
fig, axs = plt.subplots(1, len(labs), figsize=(4.6 * len(labs), 3.4), squeeze=False)
for ax, lab in zip(axs[0], labs):
    for arm in CFG.ARMS: ax.plot(PROFILES[(arm, lab)], label=arm, color=ARMC[arm], lw=1.4)
    ax.set_title(f"{lab} segments", fontsize=10); ax.set_xlabel("mel bin (low -> high frequency)"); ax.set_ylabel("share of attribution")
axs[0, 0].legend(); fig.suptitle("Where in frequency do the models look? (mean LayerCAM mass per mel bin)"); savefig("fig3_cam_frequency_profiles.png")
# 4) what GradMix does to a training sample
if "gradmix" in VIZ:
    rr = np.random.default_rng(123); fig, axs = plt.subplots(4, len(K), figsize=(4.3 * len(K), 9), squeeze=False)
    for ci_, c in enumerate(K):
        Xt, it = F[c]["train"]; ii = np.where(it == np.unique(it)[0])[0]; x = Xt[ii[len(ii) // 2]].astype(np.float32)
        cam = cam_for("gradmix", c, x[None])[0]; mg = cam_region(cam, rr, False); mr = cam_region(cam, rr, True)
        def masked(m):
            y_ = x.copy()
            if m is not None: y_[m] = float(x.mean())
            return y_
        for row, (img, ttl, cm_) in enumerate([(x, f"{c}: log-Mel", "gray"), (cam, "LayerCAM (gradmix model)", "jet"), (masked(mg), "GradMix mask applied", "gray"), (masked(mr), "area-matched random-location mask", "gray")]):
            axs[row, ci_].imshow(img, origin="lower", aspect="auto", cmap=cm_, vmin=0, vmax=1, extent=ext); axs[row, ci_].set_title(ttl, fontsize=9)
    fig.suptitle("Training-time augmentation examples"); savefig("fig4_gradmix_mask_examples.png")
# 5-7) confusion matrices, ROC, score planes (viz seed)
fig, axs = plt.subplots(1, len(CFG.ARMS), figsize=(4.2 * len(CFG.ARMS), 3.8), squeeze=False)
for ax, arm in zip(axs[0], CFG.ARMS):
    r_ = TESTS[(arm, VIZ_SEED)]; lb = r_["labels"]; ax.imshow(r_["cm"], cmap="Blues")
    ax.set_xticks(range(len(lb))); ax.set_xticklabels(lb, rotation=35, fontsize=7); ax.set_yticks(range(len(lb))); ax.set_yticklabels(lb, fontsize=7)
    for i in range(len(lb)):
        for j in range(len(lb)): ax.text(j, i, str(r_["cm"][i, j]), ha="center", va="center", fontsize=9)
    ax.set_title(arm); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
savefig("fig5_confusion_matrices.png")
fig, axs = plt.subplots(1, 2, figsize=(10, 4)); okroc = False
for arm in CFG.ARMS:
    t_ = TESTS[(arm, VIZ_SEED)]["test"]; km = (t_.true_label != "OOD").to_numpy().astype(int)
    if 0 < km.sum() < len(km):
        okroc = True
        for ax, (s, nm) in zip(axs, [(t_.L_max.to_numpy(), "Lmax"), (-t_.R_winner.to_numpy(), "-Rwinner")]):
            fpr, tpr, _ = roc_curve(km, s); ax.plot(fpr, tpr, color=ARMC[arm], label=f"{arm} (AUROC {roc_auc_score(km, s):.3f})"); ax.set_title(f"Known vs OOD ROC: {nm}")
for ax in axs: ax.plot([0, 1], [0, 1], "k--", lw=.7); ax.set_xlabel("FPR (OOD scored as known)"); ax.set_ylabel("TPR (known scored as known)"); ax.legend(fontsize=8)
if okroc: savefig("fig6_roc_by_arm.png")
else: plt.close()
fig, axs = plt.subplots(1, len(CFG.ARMS), figsize=(4.6 * len(CFG.ARMS), 4), squeeze=False)
for ax, arm in zip(axs[0], CFG.ARMS):
    r_ = TESTS[(arm, VIZ_SEED)]; t_ = r_["test"]
    for lab in r_["labels"]:
        mm = t_.true_label == lab; ax.scatter(t_.loc[mm, "L_max"], t_.loc[mm, "R_winner"], label=lab, alpha=.8, s=22)
    ax.axvline(r_["thresholds"]["tau_L"], ls="--", c="k"); ax.axhline(r_["thresholds"]["tau_R"], ls=":", c="k")
    ax.set_title(arm); ax.set_xlabel("winning calibrated score"); ax.set_ylabel("scaled Mahalanobis distance")
axs[0, 0].legend(fontsize=7); savefig("fig7_score_planes.png")
# 8) training curves
fig, axs = plt.subplots(1, len(K), figsize=(5 * len(K), 3.6), squeeze=False)
for ax, c in zip(axs[0], K):
    for arm in CFG.ARMS:
        fp = model_dir(arm, VIZ_SEED, c) / "training_log.csv"
        if fp.exists(): lg_ = pd.read_csv(fp); ax.plot(lg_.epoch, lg_.val_loss, color=ARMC[arm], label=arm)
    ax.set_title(f"{c}: validation loss"); ax.set_xlabel("epoch"); ax.set_ylabel("val loss (BCE + 0.25 MSE)")
axs[0, 0].legend(); savefig("fig8_training_curves.png")

# --------------------------------------- SAVE / ZIP -----------------------------------
summary = {"data_root": str(root), "known_classes": K, "ood_class": CFG.OOD, "arms": {a: ARM_SPEC[a] for a in CFG.ARMS}, "seeds": CFG.SEEDS,
    "viz_seed": VIZ_SEED, "wav_counts": {c: len(by_class[c]) for c in K + [CFG.OOD]}, "split_counts": {c: {p: len(v) for p, v in ss.items()} for c, ss in splits.items()},
    "gradmix_config": {k: getattr(CFG, k) for k in ["CAM_LAYER", "GM_WARMUP", "GM_PROB", "GM_AREA", "GM_FILL"]},
    "summary_table": SUMMARY.to_dict(orient="records"),
    "caveats": ["Hyper-parameters (GM_*) were fixed a priori and not tuned on tug; do not pick the best arm on the tug test and call it validated.",
                "Only a handful of tug recordings: rejection rate has a very wide confidence interval (see Wilson/bootstrap columns).",
                "One shared recording-level split; WAV-level split does not guarantee vessel- or session-level independence.",
                "Differences smaller than the seed-to-seed std are noise; use >=3 seeds before drawing conclusions.",
                "GradMix here is an adaptation (one-vs-rest per-class models, masking without label mixing); it is not a reproduction of the paper's SupCon+SSL setup."]}
(CFG.OUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
zip_path = CFG.OUT_DIR.parent / "sonar_gradmix_vs_specaug_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in CFG.OUT_DIR.rglob("*"):
        if p.is_file() and "features" not in p.parts and (CFG.ZIP_MODELS or "models" not in p.parts): zf.write(p, p.relative_to(CFG.OUT_DIR.parent))
print("\nDONE. Output folder:", CFG.OUT_DIR)
print("ZIP for download:", zip_path, "size MB:", round(zip_path.stat().st_size / 1e6, 2))
print("Key files: results_summary.md/.csv, paired_differences.csv, cam_statistics.csv, results_all_runs.csv, results/<arm>_seed*/, plots/fig1..fig8")
