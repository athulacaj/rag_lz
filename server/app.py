import sys
import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import io
# Ensure we can import from common
# Assuming server directory is at project_root/server
# We need to add project_root to sys.path
# AND common directory to sys.path because query.py uses 'from functions...' and 'from config...'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# common_dir = os.path.join(project_root, 'common')
sys.path.append(project_root)
# sys.path.append(common_dir)

import common.functions.database_utils as db_utils
from common.config import DB_NAME,EMBEDDING_MODELS,MODEL_COLLECTIONS,PARSER_LIST
from cv_agent.cv_agent_main import cv_agent_query



try:
    # We use common.query if we want to be explicit, but since common is in path, 
    # query.py's internal imports work. 
    # However, to import query_rag, we can do it from common.query
    from common.query import query_rag
except ImportError as e:
    # Try importing directly if common is in path but project_root isn't the package root
    try:
        from query import query_rag
    except ImportError as e2:
        print(f"Error importing common.query: {e}")
        print(f"Error importing query: {e2}")
        print("Make sure you are running from the correct directory or PYTHONPATH is set.")
        # Fallback to prevent immediate crash if just testing app framework
        def query_rag(q): return f"Mock response for: {q}. Error importing query_rag: {e}"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
# Allow all origins for dev simplicity
socketio = SocketIO(app, cors_allowed_origins="*")

# Custom WebSocket Logging Handler
class WebSocketHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = self.format(record)
            socketio.emit('log_message', {'data': log_entry})
        except Exception:
            pass # Handle errors if socket not ready

# Configure Logger for RAG
# We attach to the logger name 'rag_logger' which is used in common/query.py
rag_logger = logging.getLogger('rag_logger')
rag_logger.setLevel(logging.INFO)

ws_handler = WebSocketHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
ws_handler.setFormatter(formatter)
rag_logger.addHandler(ws_handler)

# Also attach to root logger or app logger if we want more logs?
# For now, just 'rag_logger' as requested: "add this log call in the query.py page"

@app.route('/config', methods=['GET'])
def get_config():
    return jsonify({
        "DB_NAME": DB_NAME,
        "EMBEDDING_MODELS": EMBEDDING_MODELS,
        "MODEL_COLLECTIONS": MODEL_COLLECTIONS,
        "PARSER_LIST": PARSER_LIST
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # We call query_rag. It logs to 'rag_logger', which emits to websocket.
    # It returns the string result.
    
    # Capture logs
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.INFO)
    rag_logger.addHandler(ch)
    
    db_name = data.get('db_name')
    embedding_model = data.get('embedding_model')
    model = data.get('model')
    parser = data.get('parser')

    try:
        answer, context_str = query_rag(question, model_name=model, embedding_model=embedding_model, parser=parser, db_name=db_name)
        
        # Save to DB
        captured_logs = log_capture_string.getvalue()
        try:
            with db_utils.get_db_connection(DB_NAME) as conn:
                db_utils.save_qa_log(conn, question, answer, captured_logs, context_str)
        except Exception as db_e:
            rag_logger.error(f"Error saving to DB: {db_e}")

        return jsonify({'response': answer})
    except Exception as e:
        rag_logger.error(f"Error in query_rag: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        rag_logger.removeHandler(ch)
        log_capture_string.close()
@app.route('/chat/cv_agent', methods=['POST'])
def chat_v2():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # We call query_rag. It logs to 'rag_logger', which emits to websocket.
    # It returns the string result.
    
    # Capture logs
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.INFO)
    rag_logger.addHandler(ch)
    
    db_name = data.get('db_name')
    embedding_model = data.get('embedding_model')
    model = data.get('model')
    parser = data.get('parser')

    try:
        answer, context_str = cv_agent_query(question, model_name=model, embedding_model=embedding_model, parser=parser, db_name=db_name)
        
        # Save to DB
        captured_logs = log_capture_string.getvalue()
        try:
            with db_utils.get_db_connection(DB_NAME) as conn:
                db_utils.save_qa_log(conn, question, answer, captured_logs, context_str)
        except Exception as db_e:
            rag_logger.error(f"Error saving to DB: {db_e}")

        return jsonify({'response': answer})
    except Exception as e:
        rag_logger.error(f"Error in query_rag: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        rag_logger.removeHandler(ch)
        log_capture_string.close()

@app.route('/history', methods=['GET'])
def get_history():
    try:
        with db_utils.get_db_connection(DB_NAME) as conn:
            history = db_utils.get_qa_history(conn)
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<int:id>', methods=['GET'])
def get_history_details(id):
    try:
        with db_utils.get_db_connection(DB_NAME) as conn:
            details = db_utils.get_qa_details(conn, id)
        if details:
            return jsonify(details)
        else:
            return jsonify({'error': 'Not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask SocketIO Server...")
    # Initialize DB tables
    with db_utils.get_db_connection(DB_NAME) as conn:
        db_utils.create_qa_tables(conn)
        
    # Use allow_unsafe_werkzeug=True if needed for dev environment with socketio
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True, use_reloader=False)
