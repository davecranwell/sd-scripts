from flask import Flask, jsonify, request

import datetime
import json  # Ensure json is imported
import os

def create_app(session_manager=None):
    app = Flask(__name__)
   
    from training_session_manager import TrainingSessionManager
    from flasgger import Swagger, swag_from

    swagger = Swagger(app)

    if session_manager is None:
        session_manager = TrainingSessionManager()

    @app.route('/training', methods=['GET'])
    @swag_from({
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
        if training is None:
            return jsonify({"error": "Training not found"}), 404
        training['epoch_losses'] = session_manager.get_epoch_losses(id)  # Fetch epoch losses
        return jsonify(dict(training))

    @app.route('/start_training', methods=['POST'])
    @swag_from({
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
            }
        }
    })
    def start_training():
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
            # Start training using the session manager
            session_id = session_manager.start_training(config)
            return jsonify({"message": "Training started", "session_id": session_id, "config": config}), 202
        except Exception as e:
            return jsonify({"error": str(e)}), 400  # Return a 400 Bad Request with the error message

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

    @app.route('/abort_training/<int:session_id>', methods=['DELETE'])
    def abort_training(session_id):
        try:
            session_manager.abort_training(session_id)
            return jsonify({"message": "Training session aborted", "session_id": session_id}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 404

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)