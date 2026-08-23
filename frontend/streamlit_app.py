import queue
import threading
import os
import tempfile
import av
import cv2
import pandas as pd
import plotly.graph_objects as go
import requests
import socketio as socketio_client
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# ============================================================
# CONFIGURATION
# ============================================================
FLASK_API_URL = "http://127.0.0.1:5000"

st.set_page_config(
    page_title="ProctorAI Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown(
    """
<style>
.main { background-color: #0E1117; }
.block-container { padding-top: 2rem; }
.dashboard-title { font-size: 32px; font-weight: 700; margin-bottom: 5px; }
.dashboard-subtitle { color: #A0A0A0; font-size: 16px; margin-bottom: 15px; }
.status-box { padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; margin-bottom: 15px; }
.medium { background-color: #493B12; color: #FACC15; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# REAL-TIME: FLASK-SOCKETIO CLIENT (Background Thread)
# ============================================================
_incident_queue = queue.Queue()
_sio_client_lock = threading.Lock()
_sio_client = None


def _ensure_socket_client():
  global _sio_client
  with _sio_client_lock:
    if _sio_client is not None:
      return

    client = socketio_client.Client(reconnection=True, reconnection_delay=2)

    @client.event
    def connect():
      print("[socketio] connected to Flask backend")

    @client.event
    def disconnect():
      print("[socketio] disconnected from Flask backend")

    @client.on("new_cheating_alert")
    def on_new_cheating_alert(data):
      _incident_queue.put(data)

    try:
      client.connect(FLASK_API_URL, wait_timeout=5)
      _sio_client = client
    except Exception as e:
      print("[socketio] connection failed:", e)


_ensure_socket_client()


def drain_new_alerts():
  alerts = []
  while not _incident_queue.empty():
    alerts.append(_incident_queue.get())
  return alerts


# ============================================================
# FLASK API HELPERS
# ============================================================


def fetch_incidents(limit=100):
  try:
    response = requests.get(
        f"{FLASK_API_URL}/api/incidents", params={"limit": limit}, timeout=5
    )
  except requests.exceptions.RequestException as error:
    print(f"[analytics] incident history request failed: {error}")
    return [], "Incident history is temporarily unavailable."

  if response.status_code != 200:
    print(
        f"[analytics] incident history returned HTTP {response.status_code}: "
        f"{response.text[:500]}"
    )
    return [], "Incident history is temporarily unavailable."

  try:
    body = response.json()
  except ValueError as error:
    print(f"[analytics] incident history returned invalid JSON: {error}")
    return [], "Incident history returned an invalid response."

  if body.get("status") != "success":
    print(f"[analytics] incident history API error: {body}")
    return [], "Incident history is temporarily unavailable."

  return body.get("data", []), None


def fetch_recordings(limit=100):
  try:
    response = requests.get(
        f"{FLASK_API_URL}/api/recordings", params={"limit": limit}, timeout=5
    )
  except requests.exceptions.RequestException as error:
    print(f"[analytics] recording history request failed: {error}")
    return [], "Recording history is temporarily unavailable."

  if response.status_code != 200:
    print(
        f"[analytics] recording history returned HTTP {response.status_code}: "
        f"{response.text[:500]}"
    )
    return [], "Recording history is temporarily unavailable."

  try:
    body = response.json()
  except ValueError as error:
    print(f"[analytics] recording history returned invalid JSON: {error}")
    return [], "Recording history returned an invalid response."

  if body.get("status") != "success":
    print(f"[analytics] recording history API error: {body}")
    return [], "Recording history is temporarily unavailable."

  return body.get("data", []), None


def analyze_video_locally(video_bytes, filename, student_id):
  suffix = os.path.splitext(filename)[1] or ".mp4"
  temp_path = None
  try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
      temp_file.write(video_bytes)
      temp_path = temp_file.name

    capture = cv2.VideoCapture(temp_path)
    if not capture.isOpened():
      return None

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = frame_count / fps if frame_count else 0
    capture.release()
    return {
      "student_id": student_id.strip() or "Unknown_Student",
        "frames_analyzed": frame_count,
        "video_frames": frame_count,
        "duration_seconds": round(duration_seconds, 2),
        "events_detected": 0,
        "event_breakdown": {},
    }
  finally:
    if temp_path:
      try:
        os.remove(temp_path)
      except OSError:
        pass


def render_snapshot(snapshot_path):
  if not snapshot_path:
    st.caption("Snapshot unavailable")
    return

  filename = snapshot_path.split("/")[-1]

  try:
    response = requests.get(
        f"{FLASK_API_URL}/static/snapshots/{filename}", timeout=5
    )
  except requests.exceptions.RequestException:
    st.caption("Snapshot unavailable")
    return

  if response.status_code == 200:
    st.image(response.content, caption=filename, use_container_width=True)
  else:
    st.caption("Snapshot unavailable")


# ============================================================
# VIDEO PROCESSOR
# ============================================================
class VideoProcessor(VideoProcessorBase):

  def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
  st.title("👁️ Monitoring")
  st.divider()

  page = st.radio("Navigation", ["Dashboard", "Analytics", "Alerts"])

  st.divider()
  st.subheader("System Status")

  if _sio_client is not None and _sio_client.connected:
    st.success("● Live updates connected")
  else:
    st.warning("● Live updates unavailable — polling every 4s")

  st.info("📹 Camera Connected")
  st.info("🤖 AI Model Active")

  st.divider()
  if st.button("🔄 Refresh incident data"):
    st.rerun()

  st.divider()
  st.caption("AI Examination Monitoring System")
  # Add this section into your Streamlit code (e.g., inside a new tab or sidebar section)

st.subheader("📤 Upload Examination Video for AI Analysis")

uploaded_video = st.file_uploader(
    "Choose a video file", type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:
  st.markdown("### Examination Video")
  st.video(uploaded_video)

student_id_input = st.text_input("Student ID", value="STUDENT_001")

if st.button("Upload & Analyze Video"):
  if uploaded_video is not None:
    video_bytes = uploaded_video.getvalue()
    normalized_student_id = student_id_input.strip() or "Unknown_Student"
    local_analysis = analyze_video_locally(
        video_bytes, uploaded_video.name, normalized_student_id
    )

    if local_analysis:
      st.subheader("Video Analytics")
      local_col1, local_col2, local_col3 = st.columns(3)
      with local_col1:
        st.metric("Video Frames", local_analysis["video_frames"])
      with local_col2:
        st.metric("Video Duration", f"{local_analysis['duration_seconds']:.1f}s")
      with local_col3:
        st.metric("Student", local_analysis["student_id"])

    with st.spinner(
        "Connecting to detection server..."
    ):
      # Prepare file payload for HTTP POST request
      files = {
          "video": (
              uploaded_video.name,
              video_bytes,
              uploaded_video.type,
          )
      }
      data = {"student_id": normalized_student_id}

      try:
        # Send video to Flask API
        response = requests.post(
          f"{FLASK_API_URL}/api/process-video", files=files, data=data, timeout=300
        )

        if response.status_code == 200:
          response_body = response.json()
          st.success(response_body.get("message", "Video analyzed successfully"))
          print(f"[analytics] upload analysis completed: {response_body}")

          analysis = response_body.get("analysis", {})
          if analysis:
            st.subheader("Analysis Results")
            result_col1, result_col2, result_col3 = st.columns(3)
            with result_col1:
              st.metric("Frames Analyzed", analysis.get("frames_analyzed", 0))
            with result_col2:
              st.metric("Events Detected", analysis.get("events_detected", 0))
            with result_col3:
              st.metric(
                  "Video Duration",
                  f"{analysis.get('duration_seconds', 0):.1f}s",
              )
            event_breakdown = analysis.get("event_breakdown", {})
            if event_breakdown:
              st.json(event_breakdown)
            else:
              st.info("No suspicious behavior was detected in this video.")

          st.markdown("### Processed Video")
          st.video(uploaded_video)

          # Automatically fetch and refresh the incident log from the database
          st.markdown("### 📥 Retrieved Answers from Database:")
          incidents, error = fetch_incidents(limit=5)
          if error:
            st.info(error)
          elif incidents:
            df_recent = pd.DataFrame(incidents)
            st.dataframe(df_recent, use_container_width=True)
          else:
            st.info("No detection results have been stored yet.")
        else:
          try:
            message = response.json().get("message", "Video analysis failed")
          except ValueError:
            message = "Video analysis failed"
          print(
              f"[analytics] upload returned HTTP {response.status_code}: "
              f"{response.text[:500]}"
          )
          st.info(message)
          local_analysis = analyze_video_locally(
              video_bytes, uploaded_video.name, normalized_student_id
          )
          if local_analysis:
            st.caption("Showing local video statistics because the backend is unavailable.")
            st.metric("Frames", local_analysis["video_frames"])
            st.metric("Duration", f"{local_analysis['duration_seconds']:.1f}s")

      except requests.exceptions.RequestException as e:
        print(f"[analytics] upload request failed: {e}")
        st.info("Video analysis is temporarily unavailable.")
        local_analysis = analyze_video_locally(
          video_bytes, uploaded_video.name, normalized_student_id
        )
        if local_analysis:
          st.caption("Showing local video statistics because the backend is unavailable.")
          local_col1, local_col2 = st.columns(2)
          with local_col1:
            st.metric("Frames", local_analysis["video_frames"])
          with local_col2:
            st.metric("Duration", f"{local_analysis['duration_seconds']:.1f}s")
  else:
    st.warning("Please select a video file first.")


# ============================================================
# LIVE UPDATES (Auto-Refresh & Toast Notifications)
# ============================================================
st_autorefresh(interval=4000, key="incident_autorefresh")

for alert in drain_new_alerts():
  st.toast(
      f"🚨 {alert.get('student_id', 'Unknown')} —"
      f" {alert.get('cheat_type', 'incident')}",
      icon="🚨",
  )

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="dashboard-title">AI-Powered Smart Examination Monitoring'
    " System</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="dashboard-subtitle">Privacy-First Human Behaviour Monitoring'
    " using Pose Estimation</div>",
    unsafe_allow_html=True,
)
st.divider()


# ============================================================
# INCIDENT LOG RENDERER
# ============================================================
def render_incident_log():
  st.subheader("📁 Previous Cheating Incidents")

  incidents, error = fetch_incidents()

  if not incidents:
    st.info("No previous detection history is available yet.")
    return

  df = pd.DataFrame(incidents)
  df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

  display_df = df.copy()
  display_df["confidence_score"] = display_df["confidence_score"].apply(
      lambda x: f"{x:.1f}%" if x is not None else "—"
  )
  display_df = display_df.rename(
      columns={
          "incident_id": "ID",
          "timestamp": "Time",
          "student_id": "Student",
          "cheat_type": "Cheat Type",
          "confidence_score": "Confidence",
          "audio_flag": "Audio Flag",
      }
  )

  st.dataframe(
      display_df[
          ["ID", "Time", "Student", "Cheat Type", "Confidence", "Audio Flag"]
      ],
      use_container_width=True,
      hide_index=True,
  )

  with st.expander("🖼️ View a Snapshot"):
    options = {
        f"#{row.incident_id} — {row.student_id} — {row.cheat_type}": row.snapshot_path
        for row in df.itertuples()
    }
    if options:
      selected = st.selectbox("Select an incident", list(options.keys()))
      render_snapshot(options[selected])


# ============================================================
# DASHBOARD PAGE
# ============================================================
if page == "Dashboard":
  incidents, _ = fetch_incidents()
  active_alerts = len(incidents) if incidents is not None else 0

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric("Students Monitored", "1")
  with col2:
    st.metric("Active Alerts", active_alerts)
  with col3:
    st.metric("Camera FPS", "30")
  with col4:
    st.metric("Current Risk", "Normal")

  st.divider()

  left, right = st.columns([2, 1])

  with left:
    st.subheader("🎥 Live Examination Feed")
    webrtc_streamer(
        key="examination-camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

  with right:
    st.subheader("🚨 Behavioural Risk")
    st.markdown(
        '<div class="status-box medium">Monitoring Active</div>',
        unsafe_allow_html=True,
    )
    st.progress(25)
    st.markdown("### Risk Score")
    st.caption("Overall Behavioural Risk Score")

  st.divider()
  render_incident_log()

# ============================================================
# ANALYTICS PAGE
# ============================================================
elif page == "Analytics":
  st.header("📊 Examination Analytics")
  incidents, error = fetch_incidents()
  recordings, recordings_error = fetch_recordings()
  df = pd.DataFrame(incidents or [])
  total_incidents = len(df)
  high_confidence = int(
      (pd.to_numeric(df["confidence_score"], errors="coerce") >= 80).sum()
  ) if not df.empty and "confidence_score" in df else 0
  avg_confidence = (
      pd.to_numeric(df["confidence_score"], errors="coerce").mean()
      if not df.empty and "confidence_score" in df else None
  )

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric("Total Events", total_incidents)
  with col2:
    st.metric("High Risk Events", high_confidence)
  with col3:
    st.metric(
        "Average Confidence",
        f"{avg_confidence:.1f}%" if avg_confidence is not None else "—",
    )
  with col4:
    st.metric("Evidence Videos", len(recordings or []))

  if not incidents and not recordings:
    if error or recordings_error:
      st.info("History is temporarily unavailable. Basic analytics will appear when PostgreSQL is connected.")
    else:
      st.info("No detection history is available yet. Analytics will appear after video detection.")
  else:
    if not df.empty:
      df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
      df["confidence_score"] = pd.to_numeric(
          df["confidence_score"], errors="coerce"
      )

      chart_col, student_col = st.columns(2)
      with chart_col:
        st.subheader("Incidents by Cheat Type")
        type_counts = df["cheat_type"].value_counts().reset_index()
        type_counts.columns = ["Cheat Type", "Events"]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=type_counts["Cheat Type"],
                y=type_counts["Events"],
                text=type_counts["Events"],
                textposition="auto",
            )
        )
        fig.update_layout(height=400, yaxis_title="Number of Events")
        st.plotly_chart(fig, use_container_width=True)

      with student_col:
        st.subheader("Incidents by Student")
        student_counts = df["student_id"].value_counts().reset_index()
        student_counts.columns = ["Student", "Events"]
        st.bar_chart(student_counts.set_index("Student"))

      st.subheader("Detection Confidence Over Time")
      confidence_df = df.dropna(subset=["timestamp", "confidence_score"])
      if not confidence_df.empty:
        st.line_chart(
            confidence_df.set_index("timestamp")["confidence_score"]
        )

        st.subheader("Recent Detection Results")
        results_df = df.copy()
        results_df["timestamp"] = results_df["timestamp"].dt.strftime(
          "%Y-%m-%d %H:%M:%S"
        )
        results_df = results_df.rename(
          columns={
            "incident_id": "ID",
            "timestamp": "Time",
            "student_id": "Student",
            "cheat_type": "Detection",
            "confidence_score": "Confidence (%)",
            "audio_flag": "Audio Flag",
          }
        )
        st.dataframe(
          results_df[
            ["ID", "Time", "Student", "Detection", "Confidence (%)", "Audio Flag"]
          ],
          use_container_width=True,
          hide_index=True,
        )

    if recordings:
      st.divider()
      st.subheader("Evidence Recordings")
      recording_options = {
          f"#{recording.get('recording_id')} — "
          f"{recording.get('student_id')} — "
          f"{recording.get('started_at', 'Unknown time')}": recording
          for recording in recordings
      }
      selected_recording = st.selectbox(
          "Select a recording", list(recording_options.keys())
      )
      recording = recording_options[selected_recording]
      video_path = recording.get("video_path", "")
      if video_path:
        filename = video_path.rstrip("/").split("/")[-1]
        st.video(f"{FLASK_API_URL}/static/recordings/{filename}")
      st.caption(
          f"Duration: {recording.get('duration_seconds', '—')} seconds"
      )

    else:
      st.info("No evidence recordings are available yet.")

# ============================================================
# ALERTS PAGE
# ============================================================
elif page == "Alerts":
  st.header("🚨 Active Alerts Log")
  incidents, error = fetch_incidents()

  if not incidents:
    st.info("No alerts have been recorded yet.")
  else:
    df = pd.DataFrame(incidents)
    df = df.sort_values("timestamp", ascending=False)
    display_df = df.rename(
        columns={
            "timestamp": "Time",
            "student_id": "Student",
            "cheat_type": "Alert",
            "confidence_score": "Confidence",
        }
    )
    st.dataframe(
        display_df[["Time", "Student", "Alert", "Confidence"]],
        use_container_width=True,
        hide_index=True,
    )
    