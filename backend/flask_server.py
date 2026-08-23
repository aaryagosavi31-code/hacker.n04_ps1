from flask import Flask, request, jsonify
import psycopg2
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================
# IMPORTANT:
# Replace these values with the SAME PostgreSQL details
# you are already using in your existing save_to_db() code.
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "Swami@15",
    "port": 5432
}


# ============================================================
# YOLO MODEL
# ============================================================
# Load the model ONCE when Flask starts.
# Do NOT load YOLO inside detect_frame().
# ============================================================

model = YOLO("ai_and_logic/yolo11n-pose.pt")


# ============================================================
# AI LOGIC
# ============================================================

def detect_frame(image):

    results = model(
        image,
        verbose=False
    )

    result = results[0]

    people_count = 0

    if result.keypoints is not None:
        people_count = len(result.keypoints)

    return {
        "people_count": people_count,
        "risk_score": 0,
        "risk_status": "LOW"
    }


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    return psycopg2.connect(**DB_CONFIG)


# ============================================================
# SAVE CHEATING INCIDENT
# ============================================================

def save_to_db(
    student_id,
    cheat_type,
    confidence_score,
    snapshot_path
):

    conn = None

    try:

        conn = get_db_connection()

        query = """
            INSERT INTO cheating_incidents
            (
                timestamp,
                student_id,
                cheat_type,
                confidence_score,
                audio_flag,
                snapshot_path
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING incident_id;
        """

        with conn.cursor() as cur:

            cur.execute(
                query,
                (
                    datetime.now(),
                    student_id,
                    cheat_type,
                    confidence_score,
                    False,
                    snapshot_path
                )
            )

            incident_id = cur.fetchone()[0]

        conn.commit()

        print(
            f"[DB SUCCESS] "
            f"Incident #{incident_id} inserted"
        )

        return incident_id

    except Exception as e:

        print(
            f"[DB ERROR] "
            f"Incident insertion failed: {e}"
        )

        if conn:
            conn.rollback()

        return None

    finally:

        if conn:
            conn.close()


# ============================================================
# GET RECENT INCIDENTS
# ============================================================

def get_recent_incidents(limit=100):

    conn = None

    try:

        conn = get_db_connection()

        query = """
            SELECT
                incident_id,
                timestamp,
                student_id,
                cheat_type,
                confidence_score,
                audio_flag,
                snapshot_path
            FROM cheating_incidents
            ORDER BY timestamp DESC
            LIMIT %s;
        """

        with conn.cursor() as cur:

            cur.execute(query, (limit,))

            columns = [
                description[0]
                for description in cur.description
            ]

            return [
                dict(zip(columns, row))
                for row in cur.fetchall()
            ]

    except Exception as e:

        print(
            f"[DB ERROR] "
            f"Could not read incidents: {e}"
        )

        return None

    finally:

        if conn:
            conn.close()


# ============================================================
# SAVE EXAMINATION RECORDING
# ============================================================
# This stores the VIDEO PATH and metadata in PostgreSQL.
# The actual .mp4 file is stored separately.
# ============================================================

def save_recording(
    student_id,
    started_at,
    ended_at,
    video_path,
    duration_seconds
):

    conn = None

    try:

        conn = get_db_connection()

        query = """
            INSERT INTO examination_recordings
            (
                student_id,
                started_at,
                ended_at,
                video_path,
                duration_seconds
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING recording_id;
        """

        with conn.cursor() as cur:

            cur.execute(
                query,
                (
                    student_id,
                    started_at,
                    ended_at,
                    video_path,
                    duration_seconds
                )
            )

            recording_id = cur.fetchone()[0]

        conn.commit()

        print(
            f"[DB SUCCESS] "
            f"Recording #{recording_id} inserted"
        )

        return recording_id

    except Exception as e:

        print(
            f"[DB ERROR] "
            f"Recording insertion failed: {e}"
        )

        if conn:
            conn.rollback()

        return None

    finally:

        if conn:
            conn.close()


# ============================================================
# GET RECENT RECORDINGS
# ============================================================

def get_recent_recordings(limit=100):

    conn = None

    try:

        conn = get_db_connection()

        query = """
            SELECT
                recording_id,
                student_id,
                started_at,
                ended_at,
                video_path,
                duration_seconds
            FROM examination_recordings
            ORDER BY started_at DESC
            LIMIT %s;
        """

        with conn.cursor() as cur:

            cur.execute(query, (limit,))

            columns = [
                description[0]
                for description in cur.description
            ]

            return [
                dict(zip(columns, row))
                for row in cur.fetchall()
            ]

    except Exception as e:

        print(
            f"[DB ERROR] "
            f"Could not read recordings: {e}"
        )

        return None

    finally:

        if conn:
            conn.close()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Flask backend is running"
    })


# ============================================================
# DETECTION API
# ============================================================
#
# Streamlit sends:
#
# POST /api/detect
#
# with:
#
# files["frame"] = JPEG image
#
# Flask:
#   JPEG → OpenCV → YOLO → result
#
# ============================================================

@app.route("/api/detect", methods=["POST"])
def detect():

    try:

        # ----------------------------------------------------
        # Check whether Streamlit sent a frame
        # ----------------------------------------------------

        if "frame" not in request.files:

            return jsonify({
                "success": False,
                "error": "No frame received"
            }), 400


        # ----------------------------------------------------
        # Read uploaded JPEG
        # ----------------------------------------------------

        file = request.files["frame"]

        image_bytes = file.read()


        # ----------------------------------------------------
        # Convert bytes → NumPy array
        # ----------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        # ----------------------------------------------------
        # Decode JPEG → OpenCV image
        # ----------------------------------------------------

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return jsonify({
                "success": False,
                "error": "Could not decode image"
            }), 400


        # ----------------------------------------------------
        # Run AI
        # ----------------------------------------------------

        result = detect_frame(image)


        # ----------------------------------------------------
        # Send result back to Streamlit
        # ----------------------------------------------------

        return jsonify({
            "success": True,
            **result
        })


    except Exception as e:

        print(
            f"[API ERROR] Detection failed: {e}"
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# INCIDENTS API
# ============================================================

@app.route("/api/incidents", methods=["GET"])
def incidents():

    try:

        limit = request.args.get(
            "limit",
            default=100,
            type=int
        )

        data = get_recent_incidents(limit)

        if data is None:

            return jsonify({
                "success": False,
                "error": "Could not retrieve incidents"
            }), 500

        return jsonify({
            "success": True,
            "incidents": data
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RECORDINGS API
# ============================================================

@app.route("/api/recordings", methods=["GET"])
def recordings():

    try:

        limit = request.args.get(
            "limit",
            default=100,
            type=int
        )

        data = get_recent_recordings(limit)

        if data is None:

            return jsonify({
                "success": False,
                "error": "Could not retrieve recordings"
            }), 500

        return jsonify({
            "success": True,
            "recordings": data
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# TEST DATABASE CONNECTION
# ============================================================

@app.route("/api/db-test", methods=["GET"])
def db_test():

    conn = None

    try:

        conn = get_db_connection()

        with conn.cursor() as cur:

            cur.execute("SELECT 1;")

            result = cur.fetchone()

        return jsonify({
            "success": True,
            "database": "connected",
            "result": result[0]
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "database": "connection_failed",
            "error": str(e)
        }), 500

    finally:

        if conn:
            conn.close()


# ============================================================
# START FLASK SERVER
# ============================================================

if __name__ == "__main__":

    print("----------------------------------------")
    print("Starting Flask Backend")
    print("----------------------------------------")
    print("Detection API : /api/detect")
    print("Incidents API : /api/incidents")
    print("Recordings API: /api/recordings")
    print("Database Test : /api/db-test")
    print("----------------------------------------")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )