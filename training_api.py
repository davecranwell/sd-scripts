from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from
import argparse  # Import argparse for command line arguments
import datetime
import json  # Ensure json is imported
import os
import threading
from io import BytesIO  # Import BytesIO for in-memory file handling

UPLOAD_FOLDER = 'uploads'

# Progress callback function
def progress_callback(loaded, total_size):
    if total_size != -1:
        print(f"Completed: {loaded} of {total_size} bytes")
    else:
        print(f"Completed: {loaded} bytes")

class UploadThread(threading.Thread):
    def __init__(self, file, output_path, progress_callback, total_length):
        super().__init__()
        self.file = file  # Keep the file object
        self.output_path = output_path
        self.progress_callback = progress_callback
        self.total_length = total_length  # Store the total length
        self._stop_event = threading.Event()

    def run(self):
        uploaded = 0

        try:
            with open(self.output_path, 'wb') as f:
                while True:
                    chunk = self.file.read(4096)  # Read in chunks
                    if not chunk or self._stop_event.is_set():
                        break

                    f.write(chunk)
                    uploaded += len(chunk)
                    self.progress_callback(uploaded, self.total_length)  # Use the stored total length
        
        except Exception as e:
            print(f"Error: {e}")

    def stop(self):
        self._stop_event.set()


def create_app(session_manager=None):
    app = Flask(__name__)
   
    upload_thread = None

    from training_session_manager import TrainingSessionManager
    

    swagger = Swagger(app, template={
        "info": {
            "title": "SD-scripts Training API",
            "description": "API for training SD SDXL Loras",
        },
    })

    if session_manager is None:
        session_manager = TrainingSessionManager()

    @app.route('/training', methods=['GET'])
    @swag_from({
        'description': 'Retrieve a list of all training sessions with epoch losses.',
        'responses': {
            200: {
                'description': 'List of all training sessions with epoch losses',
                'schema': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'integer'},
                            'start_time': {'type': 'number'},
                            'end_time': {'type': 'number'},
                            'config': {
                                'type': 'object',
                                'additionalProperties': {}
                            },
                            'current_epoch': {'type': 'integer'},
                            'current_step': {'type': 'integer'},
                            'current_loss': {'type': 'number'},
                            'last_updated': {'type': 'number'},
                            'is_completed': {'type': 'boolean'},
                            'epoch_losses': {
                                'type': 'array',
                                'items': {
                                    'type': 'object',
                                    'properties': {
                                        'epoch': {'type': 'integer'},
                                        'loss': {'type': 'number'}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })
    def get_trainings():
        trainings = session_manager.get_all_training_sessions()
        for training in trainings:
            training['epoch_losses'] = session_manager.get_epoch_losses(training['id'])  # Fetch epoch losses
        return jsonify([dict(training) for training in trainings])

    @app.route('/training/<int:id>', methods=['GET'])
    @swag_from({
        'description': 'Retrieve details of a specific training session by ID.',
        'parameters': [
            {
                'name': 'id',
                'in': 'path',
                'type': 'integer',
                'required': True,
                'description': 'The ID of the training session'
            }
        ],
        'responses': {
            200: {
                'description': 'Training session details with epoch losses',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'integer'},
                        'start_time': {'type': 'number'},
                        'end_time': {'type': 'number'},
                        'config': {
                            'type': 'object',
                            'additionalProperties': {}
                        },
                        'current_epoch': {'type': 'integer'},
                        'current_step': {'type': 'integer'},
                        'current_loss': {'type': 'number'},
                        'last_updated': {'type': 'number'},
                        'progress': {'type': 'number'},
                        'is_completed': {'type': 'boolean'},
                        'epoch_losses': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'epoch': {'type': 'integer'},
                                    'loss': {'type': 'number'}
                                }
                            }
                        }
                    }
                }
            },
            404: {
                'description': 'Training session not found'
            }
        }
    })
    def get_training(id):
        training = session_manager.get_training_session(id)
        progress = session_manager.get_total_progress(id)
        if training is None:
            return jsonify({"error": "Training not found"}), 404
        training['epoch_losses'] = session_manager.get_epoch_losses(id)  # Fetch epoch losses
        training['progress'] = progress
        return jsonify(dict(training))

    @app.route('/training', methods=['POST'])
    @swag_from({
        'description': 'Create a new training session without starting it immediately.',
        'parameters': [
            {
                'name': 'body',
                'in': 'body',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'learning_rate': {'type': 'number'},
                        'civitai_key': {'type': 'string'},  # New field for API key
                        'checkpoint_url': {'type': 'string'},  # New field for checkpoint URL
                    }
                }
            }
        ],
        'responses': {
            201: {
                'description': 'Training session created successfully',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message': {'type': 'string'},
                        'session_id': {'type': 'integer'},
                        'config': {'type': 'object'}
                    }
                }
            }
        }
    })
    def create_training_session():
        data = request.json
        
        # Define permitted keys
        permitted_keys = {
            'learning_rate',
            'civitai_key',
            'checkpoint_url'
        }
        
        # Validate incoming data
        invalid_keys = set(data.keys()) - permitted_keys
        if invalid_keys:
            return jsonify({"error": f"Invalid configuration provided: {', '.join(invalid_keys)}"}), 400

        # Load preset values from the external JSON file
        preset_config_path = os.path.join(os.path.dirname(__file__), 'preset_config.json')
        with open(preset_config_path, 'r') as f:
            preset_config = json.load(f)

        # Merge preset values with user-provided config
        config = {**preset_config, **data}
        
        # override a couple of values
        config['unet_lr'] = config['learning_rate']
        config['text_encoder_lr'] = config['learning_rate']
        
        try:
            # Create a training session using the session manager
            session_id = session_manager.create_training_session(config)
            return jsonify({"message": "Training session created", "session_id": session_id, "config": config}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 400  # Return a 400 Bad Request with the error message

    @app.route('/training/<int:id>/start', methods=['POST'])
    @swag_from({
        'description': 'Start the training for a specific training session by ID.',
        'parameters': [
            {
                'name': 'id',
                'in': 'path',
                'type': 'integer',
                'required': True,
                'description': 'The ID of the training session to start'
            }
        ],
        'responses': {
            202: {
                'description': 'Training started successfully',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message': {'type': 'string'},
                        'session_id': {'type': 'integer'},
                        'config': {'type': 'object'}
                    }
                }
            },
            404: {
                'description': 'Training session not found or invalid state'
            }
        }
    })
    def start_training(id):
        # Check if the training session exists
        training_session = session_manager.get_training_session(id)
        if training_session is None:
            return jsonify({"error": "Training session not found"}), 404

        # Check if the training session is already in progress or completed
        if training_session['is_completed']:
            return jsonify({"error": "Cannot start a completed training session."}), 400

        if session_manager.current_process is not None:
            return jsonify({"error": "Cannot start training while another session is in progress."}), 400

        try:
            # Start training using the session manager
            session_manager.start_training(training_session['config'])
            return jsonify({"message": "Training started", "session_id": id, "config": training_session['config']}), 202
        except Exception as e:
            return jsonify({"error": str(e)}), 400  # Return a 400 Bad Request with the error message

    """
    @app.route('/update_training/<int:session_id>', methods=['PUT'])
    @swag_from({
        'parameters': [
            {
                'name': 'session_id',
                'in': 'path',
                'type': 'integer',
                'required': True,
                'description': 'The ID of the training session to update'
            },
            {
                'name': 'body',
                'in': 'body',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'epoch': {'type': 'integer'},
                        'loss': {'type': 'number'}
                    }
                }
            }
        ],
        'responses': {
            200: {
                'description': 'Training session updated successfully',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message': {'type': 'string'},
                        'session_id': {'type': 'integer'}
                    }
                }
            },
            404: {
                'description': 'Training session not found'
            }
        }
    })
    def update_training(session_id):
        data = request.json
        epoch = data.get('epoch')
        loss = data.get('loss')

        # Call the update method in the session manager
        try:
            session_manager.update_training_session(session_id, epoch=epoch, loss=loss)
            return jsonify({"message": "Training session updated", "session_id": session_id}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 404
    """

    @app.route('/training/<int:id>', methods=['DELETE'])
    @swag_from({
        'description': 'Abort a training session by ID.',
        'parameters': [
            {
                'name': 'id',
                'in': 'path',
                'type': 'integer',
                'required': True,
                'description': 'The ID of the training session to abort'
            }
        ],
        'responses': {
            200: {
                'description': 'Training session aborted successfully',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message': {'type': 'string'},
                        'session_id': {'type': 'integer'}
                    }
                }
            },
            404: {
                'description': 'Training session not found'
            },
            400: {
                'description': 'Error occurred while aborting the training session'
            }
        }
    })
    def abort_training(id):
        # Check if the training session exists
        training_session = session_manager.get_training_session(id)
        if training_session is None:
            return jsonify({"error": "Training session not found"}), 404

        # Check if the training session is already completed
        if training_session['is_completed']:
            return jsonify({"error": "Cannot abort a completed training session."}), 400

        try:
            session_manager.abort_training(id)  # Abort the training session using the session manager
            return jsonify({"message": "Training session aborted", "session_id": id}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 400  # Return a 400 Bad Request with the error message

    @app.route('/training/<int:id>/upload', methods=['POST'])
    @swag_from({
        'description': 'Upload files for a specific training session by ID.',
        'parameters': [
            {
                'name': 'id',
                'in': 'path',
                'type': 'integer',
                'required': True,
                'description': 'The ID of the training session to upload files for'
            },
            {
                'name': 'file',
                'in': 'formData',
                'type': 'file',
                'required': True,
                'description': 'The file to upload'
            }
        ],
        'responses': {
            202: {
                'description': 'Upload started successfully',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'message': {'type': 'string'},
                        'filename': {'type': 'string'}
                    }
                }
            },
            400: {
                'description': 'Error occurred during file upload'
            },
            404: {
                'description': 'Training session not found or invalid state'
            }
        }
    })
    def upload_file(id):
        # Check if the training session exists
        training_session = session_manager.get_training_session(id)
        if training_session is None:
            return jsonify({"error": "Training session not found"}), 404

        # Check if the training session is already in progress or completed
        if training_session['is_completed']:
            return jsonify({"error": "Cannot upload files to a completed training session."}), 400

        if session_manager.current_process is not None:
            return jsonify({"error": "Cannot upload files while a training session is in progress."}), 400

        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Get the content length
        total_length = request.content_length

        # Create a BytesIO object to wrap the file content
        file_stream = BytesIO(file.read())  # Read the file content into memory
        file.seek(0)  # Reset the file pointer if needed
        
        # Create a directory for the training session if it doesn't exist
        session_upload_folder = os.path.join(UPLOAD_FOLDER, str(id))
        os.makedirs(session_upload_folder, exist_ok=True)

        # Save file in background thread to avoid blocking the request
        filename = secure_filename(file.filename)
        file_path = os.path.join(session_upload_folder, filename)

        # Pass the file object directly to the UploadThread for streaming
        upload_thread = UploadThread(file_stream, file_path, progress_callback, total_length)  # Pass the BytesIO object
        upload_thread.start()

        return jsonify({"message": "Upload started", "filename": filename}), 202
   
    """
    @app.route('/progress/<int:session_id>', methods=['GET'])
    @swag_from({
        'parameters': [
            {
                'name': 'session_id',
                'in': 'path',
                'type': 'integer',
                'required': True,
                'description': 'The ID of the training session'
            }
        ],
        'responses': {
            200: {
                'description': 'Training session progress',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'session_id': {'type': 'integer'},
                        'progress': {'type': 'number'}
                    }
                }
            },
            404: {
                'description': 'Training session not found'
            }
        }
    })
    def get_progress(session_id):
        try:
            progress = session_manager.get_total_progress(session_id)
            return jsonify({"session_id": session_id, "progress": progress}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 404
    """

    return app

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Start the training server.')
    parser.add_argument('--server_port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()

    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app = create_app()
    app.run(host='0.0.0.0', port=args.server_port)  # Use the port from command line argument