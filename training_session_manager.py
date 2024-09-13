import sqlite3
import time
import os
import subprocess
import threading
import json
import requests
from tqdm import tqdm
import random  # Ensure random is imported

class TrainingSessionManager:
    def __init__(self, db_connection=None):
        if db_connection:
            self.conn = db_connection
        else:
            db_path = os.path.join(os.path.dirname(__file__), 'training_history.db')
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.script_path = os.path.join(os.path.dirname(__file__), 'sdxl_train_network.py')
        self.initialize_database()
        self.current_process = None  # To keep track of the current training process
        self.current_session_id = None  # Track the current session ID
        self.lock = threading.Lock()  # Create a lock for managing training sessions

    def initialize_database(self):
        cursor = self.conn.cursor()
        
        # Create the new training_sessions table with a random ID
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_sessions (
            id INTEGER PRIMARY KEY,
            start_time REAL,
            end_time REAL,
            config TEXT,
            is_completed BOOLEAN,
            last_updated REAL,
            last_error TEXT  -- New column for storing the last error message
        )
        ''')

        # Create the epoch_losses table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS epoch_losses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            epoch INTEGER,
            loss REAL,
            FOREIGN KEY (session_id) REFERENCES training_sessions (id)
        )
        ''')
        
        self.conn.commit()

    def create_training_session(self, config):
        cursor = self.conn.cursor()
        
        # Generate a unique random ID
        session_id = random.randint(0, 2**32)
        
        # Ensure the generated ID is unique
        while cursor.execute('SELECT COUNT(*) FROM training_sessions WHERE id = ?', (session_id,)).fetchone()[0] > 0:
            session_id = random.randint(0, 2**32)

        insert_query = '''
        INSERT INTO training_sessions (
            id, start_time, config, last_updated, is_completed
        ) VALUES (?, ?, ?, ?, ?)
        '''
        
        cursor.execute(insert_query, (
            session_id,
            time.time(),
            json.dumps(config),
            time.time(),
            False
        ))
        
        self.conn.commit()

        return session_id

    def update_training_session(self, session_id, epoch=None, step=None, loss=None):
        cursor = self.conn.cursor()
        
        update_query = '''
        UPDATE training_sessions
        SET last_updated = ?
        WHERE id = ?
        '''
        
        cursor.execute(update_query, (
            time.time(),
            session_id
        ))
        
        # Record the epoch loss if provided
        if epoch is not None and loss is not None:
            self.record_epoch_loss(session_id, epoch, loss)
        
        self.conn.commit()

    def record_epoch_loss(self, session_id, epoch, loss):
        cursor = self.conn.cursor()
        
        insert_query = '''
        INSERT INTO epoch_losses (session_id, epoch, loss) VALUES (?, ?, ?)
        '''
        
        cursor.execute(insert_query, (session_id, epoch, loss))
        self.conn.commit()

    def complete_training_session(self, session_id, error_message=None):
        cursor = self.conn.cursor()
        
        update_query = '''
        UPDATE training_sessions
        SET end_time = ?, is_completed = ?, last_error = ?
        WHERE id = ?
        '''
        
        cursor.execute(update_query, (time.time(), True, error_message, session_id))
        self.conn.commit()

    def get_all_training_sessions(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.row_factory = sqlite3.Row
        cursor.execute('SELECT * FROM training_sessions')
       
        trainings = cursor.fetchall()
        return [self._process_row(row) for row in trainings]

    def get_training_session(self, session_id) -> dict | None:
        cursor = self.conn.cursor()

        cursor.row_factory = sqlite3.Row
        cursor.execute('SELECT * FROM training_sessions WHERE id = ?', (session_id,))
        training = cursor.fetchone()
        return self._process_row(training) if training else None

    def _process_row(self, row) -> dict | None:
        if row is None:
            print('row is None')
            return None
        if isinstance(row, sqlite3.Row):
            # If it's already a sqlite3.Row object, convert it to a dict
            processed = dict(row)
        elif isinstance(row, tuple):
            # If it's a tuple, create a dict using column names
            cursor = self.conn.cursor()
            columns = [column[0] for column in cursor.description]
            processed = dict(zip(columns, row))
        else:
            raise ValueError(f"Unexpected row type: {type(row)}")
        
        if 'config' in processed and processed['config']:
            try:
                processed['config'] = json.loads(processed['config'])
            except json.JSONDecodeError:
                # Handle the case where config is not valid JSON
                processed['config'] = {}
        return processed

    def run_training(self, config, session_id) -> None:
        cmd = ["python", self.script_path, "--session_id", str(session_id)] # session_id is added so train_network doesn't create its own id
        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, (int, float, str)):
                cmd.extend([f"--{key}", str(value)])
            elif isinstance(value, list):
                cmd.extend([f"--{key}"] + [str(v) for v in value])
       
        self.current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.current_process.communicate()
        
        # Check the exit code
        if self.current_process.returncode != 0:
            error_message = stderr.decode().strip()  # Capture the error message
            print(f"Error occurred during training: {error_message}")
            self.complete_training_session(session_id, error_message)  # Pass the error message
        else:
            # Handle successful completion if needed
            # we choose not to because the training script will call the method to complete the training session
            pass

    def abort_training(self, session_id):
        if self.current_process:
            self.current_process.terminate()  # Terminate the subprocess
            self.complete_training_session(session_id)  # Mark the session as completed

    def get_epoch_losses(self, session_id) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT epoch, loss FROM epoch_losses WHERE session_id = ?', (session_id,))
        losses = cursor.fetchall()
        # return [self._process_row(row) for row in losses]
        # Convert the list of tuples to a list of dictionaries
        return [{"epoch": loss[0], "loss": loss[1]} for loss in losses]

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

    def download_and_start_training(self, config, session_id):
        try:        # Download the checkpoint file in a separate thread
            civitai_key = config.get("civitai_key")  # Assume the API key is passed in the config
            checkpoint_url = config.get("checkpoint_url")  # URL for the checkpoint file
            output_path = os.path.join(config.get("output_dir"), "checkpoint_file.ckpt")  # Define the output path
            
            # Remove the civitai_key and checkpoint_url from the config as these are not known to the sd-scripts
            config.pop("civitai_key", None)
            config.pop("checkpoint_url", None)

            self.download_checkpoint(civitai_key, checkpoint_url, output_path, session_id)
            self.run_training(config, session_id)
            self.complete_training_session(session_id)
        except Exception as e:
            error_message = str(e)  # Capture the error message
            print(f"Error during download or training: {error_message}")
            self.complete_training_session(session_id, error_message)  # Pass the error message

    def download_checkpoint(self, civitai_key, checkpoint_url, output_path, session_id):
        headers = {}
        
        # Add the Authorization header only if the URL is from civitai.com
        if "civitai.com" in checkpoint_url:
            headers['Authorization'] = f'Bearer {civitai_key}'

        response = requests.get(checkpoint_url, headers=headers, stream=True)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download checkpoint: {response.text}")

        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as file, tqdm(
            desc=output_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))

    def start_training(self, config) -> int:
        with self.lock:  # Acquire the lock to prevent concurrent training starts
            if self.current_process is not None:
                raise Exception("A training session is already in progress.")  # Prevent starting a new session
            
            # Create a new training session
            session_id = self.create_training_session(config)

            # Start the training in a separate thread
            threading.Thread(target=self.download_and_start_training, args=(config, session_id)).start()
            
            return session_id