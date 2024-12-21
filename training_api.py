from flask import Flask, jsonify, request, stream_with_context, Response
from werkzeug.utils import secure_filename
from flasgger import Swagger, swag_from
import argparse  # Import argparse for command line arguments
import json  # Ensure json is imported
import os

WORKING_FOLDER_ROOT = 'runs'
UPLOAD_FOLDER = 'uploads'

def create_app(session_manager=None):
    app = Flask(__name__)
   
    from training_session_manager import TrainingSessionManager
    
    swagger_config = {
        "headers": [],
        "openapi": "3.0.2",
        "title": "SD-scripts Training API",
        "version": '',
        "termsOfService": "",
        "swagger_ui": True,
        "description": "API for training SD SDXL Loras",
    }

    Swagger(app, config=swagger_config, merge=True)

    if session_manager is None:
        session_manager = TrainingSessionManager()

    @app.route('/', methods=['GET'])
    def root():
        return "OK"

    @app.route('/training', methods=['GET'])
    @swag_from({
        'summary': 'Retrieve a list of all training sessions with epoch losses.',
        'responses': {
            200: {
                'description': 'List of all training sessions with epoch losses',
                'content': {
                    'application/json': {
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
            }
        }
    })
    def get_trainings():
        trainings = session_manager.get_all_training_sessions()
        for training in trainings:
            training['epoch_losses'] = session_manager.get_epoch_losses(training['id'])  # Fetch epoch losses
        return jsonify([dict(training) for training in trainings])

    @app.route('/training/<int:session_id>', methods=['GET'])
    @swag_from({
        'summary': 'Retrieve details of a specific training session by ID.',
        'parameters': [
            {
                'name': 'session_id',
                'in': 'path',
                'required': True,
                'description': 'The ID of the training session',
                'schema': {
                    'type': 'integer'
                }
            }
        ],
        'responses': {
            200: {
                'description': 'Training session details with epoch losses',
                'content': {
                    'application/json': {
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
                    }
                }
            },
            404: {
                'description': 'Training session not found'
            }
        }
    })
    def get_training(session_id):
        training = session_manager.get_training_session(session_id)
        progress = session_manager.get_total_progress(session_id)
        if training is None:
            return jsonify({"error": "Training session not found"}), 404

        training['epoch_losses'] = session_manager.get_epoch_losses(session_id)  # Fetch epoch losses
        training['progress'] = progress
        return jsonify(dict(training))

    @app.route('/training', methods=['POST'])
    @swag_from({
        'summary': 'Create a new training session without starting it immediately.',
        'requestBody': {
            'required': True,
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string'},
                            'webhook_url': {'type': 'string'},
                            'civitai_key': {'type': 'string'},
                            'checkpoint_url': {'type': 'string'},
                            'checkpoint_filename': {'type': 'string'}
                        }
                    }
                }
            }
        },
        'responses': {
            201: {
                'description': 'Training session created successfully',
                'content': {
                    'application/json': {
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
            }
        }
    })
    def create_training_session():
        data = request.json
        
        # Define permitted keys
        permitted_keys = {
            'id',
            'webhook_url',
            'civitai_key',
            'checkpoint_url',
            'checkpoint_filename'
        }
        
        # Validate incoming data
        invalid_keys = set(data.keys()) - permitted_keys
        if invalid_keys:
            return jsonify({"error": f"Invalid configuration provided: {', '.join(invalid_keys)}"}), 400

        try:
            # Create a training session using the session manager
            session_id = session_manager.create_training_session(data)

            return jsonify({"message": "Training session created", "session_id": session_id}), 201
        except Exception as e:
            return jsonify({"error": str(e)}), 400  # Return a 400 Bad Request with the error message

    @app.route('/training/<int:session_id>/start', methods=['POST'])
    @swag_from({
        'summary': 'Start the training for a specific training session by ID.',
        'parameters': [
            {
                'name': 'session_id',
                'in': 'path',
                'required': True,
                'description': 'The ID of the training session to start',
                'schema': {
                    'type': 'integer'
                }
            }
        ],
        'responses': {
            202: {
                'description': 'Training started successfully',
                'content': {
                    'application/json': {
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
            },
            404: {
                'description': 'Training session not found or invalid state'
            }
        }
    })
    def start_training(session_id):
        # Check if the training session exists
        training_session = session_manager.get_training_session(session_id)
        if training_session is None:
            return jsonify({"error": "Training session not found"}), 404

        try:
            # Start training using the session manager
            session_manager.start_training(training_session['id'])
            return jsonify({"message": "Training started", "session_id": session_id, "config": training_session['config']}), 202
        except Exception as e:
            return jsonify({"error": str(e)}), 400  # Return a 400 Bad Request with the error message

    @app.route('/training/<int:session_id>', methods=['DELETE'])
    @swag_from({
        'summary': 'Abort a training session by ID.',
        'parameters': [
            {
                'name': 'session_id',
                'in': 'path',
                'required': True,
                'description': 'The ID of the training session to abort',
                'schema': {
                    'type': 'integer'
                }
            }
        ],
        'responses': {
            200: {
                'description': 'Training session aborted successfully',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'message': {'type': 'string'},
                                'session_id': {'type': 'integer'}
                            }
                        }
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
    def abort_training(session_id):
        # Check if the training session exists
        training_session = session_manager.get_training_session(session_id)
        if training_session is None:
            return jsonify({"error": "Training session not found"}), 404

        try:
            session_manager.abort_training(session_id)  # Abort the training session using the session manager
            return jsonify({"message": "Training session aborted", "session_id": session_id}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 400  # Return a 400 Bad Request with the error message

    @app.route('/training/<int:session_id>/upload', methods=['POST'])
    @swag_from({
        'summary': 'Upload files for a specific training session by ID.',
        'requestBody': {
            'required': True,
            'content': {
                'multipart/form-data': {
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'files': {
                                'type': 'array',
                                'items': {
                                    'type': 'string',
                                    'format': 'binary'
                                }
                            }
                        }
                    }
                }
            }
        },
        'parameters': [
            {
                'name': 'session_id',
                'in': 'path',
                'required': True,
                'description': 'The ID of the training session to upload files for',
                'schema': {
                    'type': 'integer'
                }
            },
        ],
        'responses': {
            200: {
                'description': 'Upload completed successfully',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'message': {'type': 'string'},
                                'filenames': {
                                    'type': 'array',
                                    'items': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
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
    def upload_file(session_id):
        training_session = session_manager.get_training_session(session_id)
        if training_session is None:
            return jsonify({"error": "Training session not found"}), 404

        if training_session['is_completed']:
            return jsonify({"error": "Cannot upload files to a completed training session."}), 400

        if 'files' not in request.files:
            return jsonify({"error": "No file part"}), 400

        files = request.files.getlist('files')
        if len(files) == 0 or len(files) > 100:
            return jsonify({"error": "You must upload between 1 and 100 files."}), 400

        def handle_upload():
            uploaded_filenames = []
            for file in files:
                if file.filename == '':
                    yield json.dumps({"error": "One or more files have no selected filename."}) + '\n'
                    return

                filename = secure_filename(file.filename)
                file_path = os.path.join(training_session['config']['train_data_dir'], filename)

                try:
                    with open(file_path, 'wb') as f:
                        chunk_size = 4096
                        uploaded = 0
                        while True:
                            chunk = file.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            uploaded += len(chunk)
                            yield json.dumps({
                                "filename": filename,
                                "uploaded": uploaded,
                                "total": file.content_length
                            }) + '\n'
                    uploaded_filenames.append(filename)
                except Exception as e:
                    yield json.dumps({"error": f"Error uploading {filename}: {str(e)}"}) + '\n'

            yield json.dumps({
                "message": "Upload completed",
                "filenames": uploaded_filenames
            }) + '\n'

        return Response(stream_with_context(handle_upload()), content_type='application/json')

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