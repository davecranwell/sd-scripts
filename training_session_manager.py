import sqlite3
import time
import os
import subprocess
import threading
import json
import random  # Ensure random is imported
import pycurl  # Import pycurl for downloading files
import shutil  # Import shutil for checking disk space
import certifi
import requests
from utils import AbortableThread  # Import AbortableThread from utils

WORKING_FOLDER_ROOT = 'runs'
PRETRAINED_MODEL_PATH = 'pretrained'

class TrainingSessionManager:
    def __init__(self, db_connection=None):
        if db_connection:
            self.conn = db_connection
        else:
            db_path = os.path.join(os.path.dirname(__file__), 'training_history.db')
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.script_path = os.path.join(os.path.dirname(__file__), 'sdxl_train_network.py')
        self.initialize_database()
        self.training_process = None  # To keep track of the current training process
        self.current_session_id = None  # Track the current session ID
        self.lock = threading.Lock()  # Create a lock for managing training sessions
        self.download_progress = 0  # Class variable to store download progress percentage
        self.training_thread = None

    def initialize_database(self):
        cursor = self.conn.cursor()
        
        # Create the new training_sessions table with a random ID
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_sessions (
            id INTEGER PRIMARY KEY,
            start_time REAL,
            remaining INTEGER,
            status TEXT,
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
        
        # Load preset values from the external JSON file
        preset_config_path = os.path.join(os.path.dirname(__file__), 'preset_config.json')
        with open(preset_config_path, 'r') as f:
            preset_config = json.load(f)

        # Merge preset values with user-provided config
        config = {**preset_config, **config}

        # Generate a unique random ID
        session_id = config.get('id', random.randint(0, 2**32))

        # Modify paths in the config with the session_id
        config['output_dir'] = os.path.join(WORKING_FOLDER_ROOT, str(session_id), config['output_dir'])
        config['logging_dir'] = os.path.join(WORKING_FOLDER_ROOT, str(session_id), config['logging_dir'])
        config['train_data_dir'] = os.path.join(WORKING_FOLDER_ROOT, str(session_id), config['train_data_dir'])
        # Uncomment if you want to modify pretrained_model_name_or_path as well
        # config['pretrained_model_name_or_path'] = os.path.join(WORKING_FOLDER_ROOT, str(session_id), config['pretrained_model_name_or_path'])

        # Create necessary directories
        for dir_path in [config['output_dir'], config['logging_dir'], config['train_data_dir'], config['pretrained_model_name_or_path']]:
            os.makedirs(dir_path, exist_ok=True)
        
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

    def update_training_session(self, session_id, **kwargs):
        cursor = self.conn.cursor()
        
        # Build dynamic update query based on kwargs
        update_fields = ['last_updated = ?']
        params = [time.time()]

        if 'status' in kwargs:
            update_fields.append('status = ?')
            params.append(kwargs['status'])
        
        if 'remaining' in kwargs:
            update_fields.append('remaining = ?')
            params.append(kwargs['remaining'])
        
        update_query = f'''
        UPDATE training_sessions
        SET {', '.join(update_fields)}
        WHERE id = ?
        '''
        
        # append session_id last so it can be used in the WHERE clause
        params.append(session_id)
        cursor.execute(update_query, tuple(params))
        
        # Record the epoch loss if provided
        if 'epoch' in kwargs and 'loss' in kwargs:
            self.record_epoch_loss(session_id, kwargs['epoch'], kwargs['loss'])

        # Fire webhook call with an object that is passed directly to the webhook post body
        self.fire_webhook(session_id)

        # If new_config is provided, update the config field
        if 'new_config' in kwargs:
            training_session = self.get_training_session(session_id)
            if training_session:
                existing_config = training_session['config']
                existing_config.update(kwargs['new_config'])  # Merge new properties into existing config
                cursor.execute('''
                    UPDATE training_sessions
                    SET config = ?
                    WHERE id = ?
                ''', (json.dumps(existing_config), session_id))  # Convert to JSON string before storing

       
        self.conn.commit()

    def fire_webhook(self, session_id, data):
        # TODO: Implement webhook firing
        training_session = self.get_training_session(session_id)
        if training_session and 'webhook_url' in training_session['config']:
            webhook_url = training_session['config']['webhook_url']
            
            # Call webhook with GET request
            # convert training_session to json
            training_session_json = json.dumps(training_session)
            requests.post(webhook_url, json=training_session_json)

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
        SET end_time = ?, is_completed = ?, last_error = ?, status = ?
        WHERE id = ?
        ''' 
    
        cursor.execute(update_query, (time.time(), error_message is None, error_message, "complete", session_id))
        self.conn.commit()

        # Fire webhook call
        self.fire_webhook(session_id)

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

        self.training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.training_process.communicate()
        
        # Fire webhook call
        self.fire_webhook(session_id)

        # Check the exit code
        if self.training_process.returncode != 0:
            error_message = stderr.decode().strip()  # Capture the error message
            print(f"Error occurred during training: {error_message}")
            self.complete_training_session(session_id, error_message)  # Pass the error message
        else:
            self.complete_training_session(session_id)

    def abort_training(self, session_id):
        training_session = self.get_training_session(session_id)

         # Check if the training session is already completed
        if training_session['is_completed']:
            raise Exception(f"Cannot abort a completed training session.")

        if self.training_process:
            self.training_process.terminate()  # Terminate the subprocess if runnin

        if self.training_thread is not None and self.training_thread.is_alive():
            self.training_thread.stop()  # Stop the training thread
        
        self.complete_training_session(session_id, "Training aborted")  # Mark the session as completed

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

    def download_and_run(self, config, session_id):
        try:        # Download the checkpoint file in a separate thread
            civitai_key = config.get("civitai_key")  # Assume the API key is passed in the config
            checkpoint_url = config.get("checkpoint_url")  # URL for the checkpoint file
            checkpoint_filename = config.get("checkpoint_filename")
            output_path = config.get("pretrained_model_name_or_path")
            
            # Remove the civitai_key and checkpoint_url from the config as these are not known to the sd-scripts
            config.pop("civitai_key", None)
            config.pop("checkpoint_url", None)

            downloaded_images = self.download_images(session_id)
            
            downloaded_checkpoint = self.download_checkpoint(civitai_key, checkpoint_url, output_path, session_id, checkpoint_filename)
           
            if downloaded_images and downloaded_checkpoint:
                self.run_training(config, session_id)
        except Exception as e:
            error_message = str(e)  # Capture the error message
            print(f"Error during download and training: {error_message}")
            self.complete_training_session(session_id, error_message)  # Pass the error message


    def download_images(self, session_id):
        training_session = self.get_training_session(session_id)
        if not training_session or 'config' not in training_session:
            raise Exception("Training session not found or config is missing")

        config = training_session['config']
        images_url = config.get('images_url')
        if not images_url:
            raise Exception("No images URL provided in config")

        train_data_dir = config.get('train_data_dir')
        if not train_data_dir:
            raise Exception("No train_data_dir specified in config")

        # Download to a temporary zip file
        zip_path = os.path.join(train_data_dir, 'images.zip')
        
        self._download_file(
            url=images_url,
            output_path=zip_path,
            session_id=session_id,
            status_prefix="downloading_images"
        )
        
        # Unzip the downloaded file
        shutil.unpack_archive(zip_path, train_data_dir)
        os.remove(zip_path)  # Clean up zip file after extraction
        
        return True

    def _download_file(self, url, output_path, headers=None, session_id=None, status_prefix="downloading"):
        """
        Generic file download function using pycurl
        
        Args:
            url (str): URL to download from
            output_path (str): Full path where to save the file
            headers (list, optional): List of HTTP headers
            session_id (int, optional): Training session ID for progress updates
            status_prefix (str, optional): Prefix for status updates (e.g., "downloading_checkpoint")
        
        Returns:
            bool: True if download completed successfully, False if aborted
        """
        if session_id:
            self.update_training_session(session_id, status=f"{status_prefix}_started")
            self.fire_webhook(session_id)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize curl
        curl = pycurl.Curl()
        curl.setopt(curl.URL, url)
        if headers:
            curl.setopt(curl.HTTPHEADER, headers)
        curl.setopt(curl.CAINFO, certifi.where())
        curl.setopt(curl.VERBOSE, True)
        curl.setopt(curl.FOLLOWLOCATION, True)
        curl.setopt(pycurl.USERAGENT, b"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")

        with open(output_path, 'wb') as file:
            self.download_progress = 0

            def progress_callback(total_to_download, downloaded, total_to_upload, uploaded):
                if total_to_download > 0:
                    self.download_progress = (downloaded / total_to_download) * 100
                return 1 if self.training_thread._stop_event.is_set() else 0

            curl.setopt(curl.WRITEDATA, file)
            curl.setopt(curl.NOPROGRESS, False)
            curl.setopt(curl.XFERINFOFUNCTION, progress_callback)

            try:
                curl.perform()
                response_code = curl.getinfo(curl.RESPONSE_CODE)
                if response_code != 200:
                    raise Exception(f"Failed to download file: HTTP {response_code}")

                total_size = int(curl.getinfo(curl.CONTENT_LENGTH_DOWNLOAD))
                
                # Check available disk space
                total, used, free = shutil.disk_usage(os.path.dirname(output_path))
                if free < total_size:
                    raise Exception(f"Not enough disk space. Required: {total_size}, Available: {free}")

                print(f"Downloaded {total_size} bytes to {output_path}")
                self.download_progress = 100

                if session_id:
                    self.update_training_session(session_id, status=f"{status_prefix}_completed")
                    self.fire_webhook(session_id)
                    
                return True
            except Exception as e:
                print(f"Error during download: {e}")
                if session_id:
                    self.update_training_session(session_id, status=f"{status_prefix}_failed")
                    self.fire_webhook(session_id)
                raise
            finally:
                curl.close()

    def download_checkpoint(self, civitai_key, checkpoint_url, output_path, session_id, checkpoint_filename):
        headers = []
        
        if "civitai.com" in checkpoint_url.lower():
            headers.append(f'Authorization: Bearer {civitai_key}')
            checkpoint_url = checkpoint_url + "&token=" + civitai_key

        file_path = os.path.join(output_path, checkpoint_filename)
       
        return self._download_file(
            url=checkpoint_url,
            output_path=file_path,
            headers=headers,
            session_id=session_id,
            status_prefix="downloading_checkpoint"
        )


    def start_training(self, session_id) -> int:
        # Get the training config
        training_session = self.get_training_session(session_id)

        # Check if the training session is already in progress or completed
        if training_session['is_completed']:
            raise Exception(f"Cannot start a completed training session")

        if self.training_process is not None:
            raise Exception(f"Cannot start training while another session is in progress.")

        if training_session is None or 'config' not in training_session:
            raise Exception("Training session not found or config is missing.")

        config = training_session['config']

        with self.lock:  # Acquire the lock to prevent concurrent training starts
            if self.training_thread is not None and self.training_thread.is_alive():
                raise Exception("A training session is already in progress.")  # Prevent starting a new session

            # Start the training in a separate thread
            self.training_thread = AbortableThread(self.download_and_run, config, session_id)
            self.training_thread.start()
            
            return session_id

    def get_total_progress(self, session_id) -> float:
        # Get the total number of epochs from the training config
        training_session = self.get_training_session(session_id)
        if training_session is None or 'config' not in training_session:
            return 0.0  # Return 0 if the session is not found or config is missing

        total_epochs = training_session['config'].get('epoch', 0)  # Get total epochs from config
        if total_epochs <= 0:
            return 0.0  # Return 0 if there are no epochs to train

        # Get the highest epoch number completed from the epoch_losses table
        cursor = self.conn.cursor()
        cursor.execute('SELECT MAX(epoch) FROM epoch_losses WHERE session_id = ?', (session_id,))
        highest_epoch = cursor.fetchone()[0] or 0  # Default to 0 if no epochs found

        # Calculate training progress
        training_progress = (highest_epoch / total_epochs) * 100 if total_epochs > 0 else 0

        # Combine download progress and training progress
        overall_progress = (self.download_progress + training_progress) / 2  # Average of both progress

        return overall_progress