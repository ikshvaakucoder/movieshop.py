# app.py
# Streamlit prototype for: 2D->3D character, mocap from video, manual per-frame bone fix, export to Blender for "realify".
# pip install streamlit mediapipe opencv-python numpy pillow scipy trimesh open3d pydantic

import io
import os
import json
import time
import zipfile
import tempfile
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from dataclasses import dataclass, asdict

# --------- Optional: MediaPipe Pose for markerless mocap ----------
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False

# ------------------- Simple data structures -----------------------

LANDMARK_NAMES = [
    # MediaPipe Pose 33 keypoints (indices preserved)
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index"
]

@dataclass
class FramePose:
    frame_idx: int
    width: int
    height: int
    # (N,2) normalized in [0,1], and visibility (N,)
    xy: list
    vis: list

    def to_dict(self):
        return asdict(self)

# ------------------- Utilities -----------------------

def extract_frames(video_bytes, every_n=1, max_frames=600):
    """Read video bytes -> list of (idx, frame_bgr)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        path = f.name
    cap = cv2.VideoCapture(path)
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % every_n == 0:
            frames.append((i, frame))
        if len(frames) >= max_frames: break
        i += 1
    cap.release()
    return frames

def pose_from_frame(frame_bgr, pose):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if not result.pose_landmarks:
        return None
    xy = []
    vis = []
    for lm in result.pose_landmarks.landmark:
        xy.append([float(lm.x), float(lm.y)])
        vis.append(float(lm.visibility))
    return np.array(xy), np.array(vis), (w, h)

def draw_skeleton(frame_bgr, xy_norm, color=(0,255,0)):
    h, w = frame_bgr.shape[:2]
    pts = (xy_norm * np.array([w, h])).astype(int)
    out = frame_bgr.copy()
    # quick bones (shoulders/arms/hips/legs); add more if you like
    pairs = [
        (11,13),(13,15),   # left arm
        (12,14),(14,16),   # right arm
        (11,12),          # shoulders
        (23,24),          # hips
        (11,23),(12,24),  # torso
        (23,25),(25,27),  # left leg
        (24,26),(26,28),  # right leg
        (27,29),(28,30),  # ankles->heels
        (29,31),(30,32)   # heels->feet
    ]
    for (a,b) in pairs:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, 2)
    for p in pts:
        cv2.circle(out, tuple(p), 3, (255,0,0), -1)
    return out

def save_animation_json(frame_poses, save_path):
    data = [fp.to_dict() for fp in frame_poses]
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

# ------------------- Streamlit UI -----------------------

st.set_page_config(page_title="2D→3D De-Age + Mocap + Realify (Prototype)", layout="wide")
st.title("🧠🎭 2D→3D De-Age + Mocap + Realify — Prototype (Python)")

st.markdown("""
This demo lets you:
1) Upload a **younger/older face photo** (for reference) and an **acting video** (kid/adult acting).
2) Run **markerless motion capture** (auto bones).
3) **Correct bones frame-by-frame** if needed.
4) Export an **animation package** to “realify” in Blender (photoreal render).
""")

colL, colR = st.columns([1,1])

with colL:
    face_img = st.file_uploader("Younger/Older Face Photo (PNG/JPG)", type=["png","jpg","jpeg"])
    video = st.file_uploader("Acting Video (MP4/MOV)", type=["mp4","mov","m4v","avi"])
    every_n = st.slider("Sample every Nth frame", 1, 10, 2, 1)
    run_mocap = st.button("Run Auto Mocap (MediaPipe)")

with colR:
    st.info("Tip: If MediaPipe is not installed, install it with `pip install mediapipe`.")
    if not MP_AVAILABLE:
        st.warning("MediaPipe not available in this environment. The code still runs, but mocap step will be skipped.")

# Session state
if "frames" not in st.session_state: st.session_state.frames = []
if "poses" not in st.session_state: st.session_state.poses = []
if "wh" not in st.session_state: st.session_state.wh = None

# ---------- Extract frames ----------
if video is not None and st.session_state.frames == []:
    with st.spinner("Extracting frames..."):
        vbytes = video.read()
        frames = extract_frames(vbytes, every_n=every_n, max_frames=800)
        st.session_state.frames = frames
    st.success(f"Loaded {len(st.session_state.frames)} frames.")

# ---------- Auto mocap ----------
if run_mocap and st.session_state.frames:
    if not MP_AVAILABLE:
        st.error("MediaPipe not installed. Please install mediapipe to run mocap.")
    else:
        with st.spinner("Running markerless mocap on sampled frames..."):
            mp_pose = mp.solutions.pose
            poses = []
            wh = None
            with mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, smooth_landmarks=True) as pose:
                for idx, frame in st.session_state.frames:
                    res = pose_from_frame(frame, pose)
                    if res is None:
                        continue
                    xy, vis, (w,h) = res
                    if wh is None: wh = (w,h)
                    poses.append(FramePose(
                        frame_idx=idx, width=w, height=h,
                        xy=xy.tolist(), vis=vis.tolist()
                    ))
            st.session_state.poses = poses
            st.session_state.wh = wh
        st.success(f"Mocap completed on {len(st.session_state.poses)} frames.")

# ---------- Pose editor ----------
if st.session_state.poses:
    st.subheader("🧩 Frame-by-frame Bone Correction")
    idx_options = [fp.frame_idx for fp in st.session_state.poses]
    sel = st.selectbox("Select a frame to edit", idx_options, index=min(5, len(idx_options)-1))
    fp = next(p for p in st.session_state.poses if p.frame_idx == sel)
    xy = np.array(fp.xy, dtype=np.float32)
    vis = np.array(fp.vis, dtype=np.float32)
    w, h = fp.width, fp.height

    # show image with skeleton
    frame_bgr = next(frm for (i, frm) in st.session_state.frames if i == fp.frame_idx)
    overlay = draw_skeleton(frame_bgr, xy)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=f"Frame {fp.frame_idx} — click below to adjust")

    # quick editor for a few key joints; expand to all if you want
    with st.expander("Edit joints"):
        joint_names_to_edit = ["left_shoulder","right_shoulder","left_elbow","right_elbow",
                               "left_wrist","right_wrist","left_hip","right_hip",
                               "left_knee","right_knee","left_ankle","right_ankle","nose"]
        for name in joint_names_to_edit:
            j = LANDMARK_NAMES.index(name)
            c1, c2, c3 = st.columns(3)
            with c1:
                x = st.slider(f"{name} • x", 0.0, 1.0, float(xy[j,0]), 0.001, key=f"x_{name}")
            with c2:
                y = st.slider(f"{name} • y", 0.0, 1.0, float(xy[j,1]), 0.001, key=f"y_{name}")
            with c3:
                v = st.slider(f"{name} • visibility", 0.0, 1.0, float(vis[j]), 0.01, key=f"v_{name}")
            xy[j,0], xy[j,1], vis[j] = x, y, v

    if st.button("Save corrections to this frame"):
        fp.xy = xy.tolist()
        fp.vis = vis.tolist()
        st.success("Saved corrections.")

    # preview corrected overlay
    overlay2 = draw_skeleton(frame_bgr, xy, color=(0,200,255))
    st.image(cv2.cvtColor(overlay2, cv2.COLOR_BGR2RGB), caption="Corrected skeleton preview")

# ---------- Retarget + Export ----------
st.subheader("🧵 Retarget & Export")

st.markdown("""
This prototype exports:
- `animation.json` — per-frame 2D keypoints (normalized) after your corrections
- `reference_face.png` — your uploaded younger/older face
- `frames_preview/` — a few overlay frames for sanity check

**Next step**: run `blender --background --python blender_realify.py` to import your rig/mesh, build an armature, retarget motion, and render photoreal (Cycles).
""")

if st.button("Build Export Package"):
    if not st.session_state.poses:
        st.error("No poses available. Run mocap first.")
    else:
        with tempfile.TemporaryDirectory() as td:
            anim_json = os.path.join(td, "animation.json")
            save_animation_json(st.session_state.poses, anim_json)

            # save reference face if provided
            ref_face_path = None
            if face_img is not None:
                ref = Image.open(face_img).convert("RGB")
                ref_face_path = os.path.join(td, "reference_face.png")
                ref.save(ref_face_path)

            # dump a few overlays
            prev_dir = os.path.join(td, "frames_preview")
            os.makedirs(prev_dir, exist_ok=True)
            for k, fp in enumerate(st.session_state.poses[:10]):  # first 10
                frame_bgr = next(frm for (i, frm) in st.session_state.frames if i == fp.frame_idx)
                ov = draw_skeleton(frame_bgr, np.array(fp.xy))
                cv2.imwrite(os.path.join(prev_dir, f"overlay_{fp.frame_idx:05d}.png"), ov)

            # pack zip
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(anim_json, "animation.json")
                if ref_face_path:
                    z.write(ref_face_path, "reference_face.png")
                for f in sorted(os.listdir(prev_dir)):
                    z.write(os.path.join(prev_dir, f), f"frames_preview/{f}")
            buf.seek(0)
            st.download_button("Download package (animation_export.zip)", data=buf, file_name="animation_export.zip", mime="application/zip")
            st.success("Export ready. Use it with blender_realify.py")

