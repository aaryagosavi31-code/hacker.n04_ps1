import psycopg2
from datetime import datetime

DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'Swami@15',
    'host': 'localhost',
    'port': '5432'
}

def save_to_db(student_id, cheat_type, confidence_score, snapshot_path):
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        query = """
            INSERT INTO cheating_incidents 
            (timestamp, student_id, cheat_type, confidence_score, audio_flag, snapshot_path)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING incident_id;
        """
        with conn.cursor() as cur:
            cur.execute(query, (
                datetime.now(), student_id, cheat_type, 
                confidence_score, False, snapshot_path
            ))
            incident_id = cur.fetchone()[0]
        
        conn.commit()
        print(f"[DB SUCCESS] Successfully inserted Incident ID #{incident_id} into PostgreSQL!")
        return incident_id

    except Exception as e:
        print(f"[DB ERROR] Insertion failed: {e}")
        if conn:
            conn.rollback()
        return None

    finally:
        if conn:
            conn.close()