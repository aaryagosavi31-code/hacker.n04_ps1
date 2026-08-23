import os
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
from db import save_to_db, get_recent_incidents

app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Serve snapshots to the React frontend
@app.route('/static/snapshots/<filename>')
def serve_snapshot(filename):
    return send_from_directory('static/snapshots', filename)

# Endpoint called by ai_and_logic when a cheating incident occurs
@app.route('/api/incident', methods=['POST'])
def handle_incident():
    data = request.json
    student_id = data.get('student_id')
    cheat_type = data.get('cheat_type')
    confidence_score = data.get('confidence_score')
    snapshot_path = data.get('snapshot_path')

    # Save to PostgreSQL without bench_id
    incident_id = save_to_db(student_id, cheat_type, confidence_score, snapshot_path)

    if incident_id is None:
        return jsonify({'status': 'error', 'message': 'Failed to save to database'}), 500

    payload = {
        'incident_id': incident_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'student_id': student_id,
        'cheat_type': cheat_type,
        'confidence_score': confidence_score,
        'snapshot_path': snapshot_path
    }

    # Broadcast real-time update to React frontend
    socketio.emit('new_cheating_alert', payload)
    return jsonify({'status': 'success', 'data': payload}), 201


@app.route('/api/incidents', methods=['GET'])
def list_incidents():
    try:
        limit = min(int(request.args.get('limit', 100)), 500)
    except ValueError:
        limit = 100

    incidents = get_recent_incidents(limit)
    if incidents is None:
        return jsonify({'status': 'error', 'message': 'Could not read incident logs'}), 500

    for incident in incidents:
        if incident.get('timestamp') is not None:
            incident['timestamp'] = incident['timestamp'].isoformat(sep=' ')

    return jsonify({'status': 'success', 'data': incidents}), 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
   
   
@app.route('/api/process-video', methods=['POST'])
def process_video_endpoint():
  # 1. Retrieve the file and form data from the request object first
  if 'video' not in request.files:
    return jsonify({'status': 'error', 'message': 'No video file provided'}), 400

  video_file = request.files['video']
  student_id = request.form.get('student_id', 'Unknown_Student')

  # 2. Now 'video_file' and 'student_id' are defined and ready to use
  upload_path = os.path.join('uploads', video_file.filename)
  os.makedirs('uploads', exist_ok=True)
  video_file.save(upload_path)

  # Example save to database call
  sample_cheat = 'Unauthorized Object (Phone)'
  sample_confidence = 88.5
  incident_id = save_to_db(
      student_id, sample_cheat, sample_confidence, 'snapshot_sample.jpg'
  )

  return jsonify({
      'status': 'success',
      'message': 'Video processed successfully',
      'incident_id': incident_id,
  })