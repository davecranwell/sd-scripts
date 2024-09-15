import pytest
import time
import json
import sqlite3
from flask import Flask
from training_api import create_app
from training_session_manager import TrainingSessionManager
from unittest.mock import patch, MagicMock
import subprocess
import requests

@pytest.fixture(autouse=True)
def mock_download_and_run():
    with patch.object(TrainingSessionManager, 'download_and_run') as mock:
        yield mock

@pytest.fixture(scope='module')
def in_memory_db():
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()

@pytest.fixture(scope='module')
def session_manager(in_memory_db):
    manager = TrainingSessionManager(db_connection=in_memory_db)
    yield manager

@pytest.fixture
def client(session_manager):
    training_api_app = create_app(session_manager)
    training_api_app.config['TESTING'] = True
    with training_api_app.test_client() as client:
        yield client

def test_create_training_session(mock_download_and_run, client, session_manager):
    mock_download_and_run.return_value = None

    response = client.post('/start_training', json={
        "checkpoint_url": "https://civitai.com/checkpoint"
    })

    assert response.status_code == 202
    mock_download_and_run.assert_called_once()

    data = json.loads(response.data)
    session_id = data['session_id']
    training_session = session_manager.get_training_session(session_id)

    assert training_session is not None
    assert training_session['config']['checkpoint_url'] == "https://civitai.com/checkpoint"

def test_update_training_session(client, session_manager):
    session_id = session_manager.create_training_session({
        "checkpoint_url": "https://civitai.com/checkpoint"
    })
   
    session_manager.update_training_session(session_id, epoch=1, loss=0.05)

    updated_session = session_manager.get_training_session(session_id)
    assert updated_session is not None
    assert updated_session['last_updated'] is not None
   
    epoch_losses = session_manager.get_epoch_losses(session_id)

    assert len(epoch_losses) == 1
    assert epoch_losses[0]['epoch'] == 1
    assert epoch_losses[0]['loss'] == 0.05

def test_complete_training_session(client, session_manager):
    session_id = session_manager.create_training_session({
        "checkpoint_url": "https://civitai.com/checkpoint"
    })

    session_manager.complete_training_session(session_id)

    completed_session = session_manager.get_training_session(session_id)
    assert completed_session is not None
    assert completed_session['is_completed'] is 1

# def test_run_training_error_handling(mock_download_and_run, client, session_manager):
#     # Create a mock for the subprocess.Popen
#     mock_process = MagicMock()
#     mock_process.communicate.return_value = (b'output', b'error')  # Simulate output and error
#     mock_process.returncode = 1  # Simulate a non-zero exit code
#     mock_download_and_run.return_value = mock_process  # Mock Popen to return the mock process

#     # Call the start_training method
#     session_id = session_manager.start_training({
#         "pretrained_model_name_or_path": "path/to/model",
#         "train_data_dir": "path/to/train/data",
#         "output_dir": "path/to/output",
#         "resolution": 1024,
#         "train_batch_size": 1,
#         "learning_rate": 1e-5,
#         "max_train_steps": 1000,
#         "network_module": "networks.lora",
#         "network_dim": 32,
#         "network_alpha": 16,
#     })

#     # frankly ridiculous but it works
#     time.sleep(2) 

#     mock_download_and_run.assert_called_once()
 
#     # Fetch the completed training session
#     completed_session = session_manager.get_training_session(session_id)
#     assert completed_session is not None
#     assert completed_session['is_completed'] is 1  # Check if the session is marked as completed

@patch('requests.get')
def test_download_checkpoint_with_auth(mock_get, session_manager):
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'content-length': '100'}

    # Simulate the iter_content method
    def iter_content(chunk_size=1):
        yield b'Test content part 1'
        yield b'Test content part 2'

    mock_response.iter_content = iter_content
    mock_get.return_value = mock_response

    # Test with civitai.com URL
    civitai_key = "test_civitai_key"
    checkpoint_url = "https://civitai.com/checkpoint"
    output_path = "test_checkpoint.ckpt"
    session_id = 1  # Example session ID

    # Call the download function
    session_manager.download_checkpoint(civitai_key, checkpoint_url, output_path, session_id)

    # Check that the request was made with the correct headers
    mock_get.assert_called_once_with(checkpoint_url, headers={'Authorization': f'Bearer {civitai_key}'}, stream=True)


@patch('requests.get')
def test_download_checkpoint_without_auth(mock_get, session_manager):
    # Create a mock response object
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'content-length': '100'}

    # Simulate the iter_content method
    def iter_content(chunk_size=1):
        yield b'Test content part 1'
        yield b'Test content part 2'

    mock_response.iter_content = iter_content
    mock_get.return_value = mock_response

    # Test with a different URL
    civitai_key = "test_civitai_key"
    checkpoint_url = "https://otherdomain.com/checkpoint"
    output_path = "test_checkpoint.ckpt"
    session_id = 1  # Example session ID

    # Call the download function
    session_manager.download_checkpoint(civitai_key, checkpoint_url, output_path, session_id)

    # Check that the request was made without the Authorization header
    mock_get.assert_called_once_with(checkpoint_url, headers={}, stream=True)

def test_api_responsiveness_during_download(mock_download_and_run, client, session_manager):
    # Mock the download_checkpoint to simulate a long-running download
    def mock_download(*args, **kwargs):
        time.sleep(1)  # Simulate a long download time
    
    mock_download_and_run.side_effect = mock_download

    # Start the training session, which will trigger the download
    response = client.post('/start_training', json={
        "checkpoint_url": "https://civitai.com/checkpoint"
    })


    assert response.status_code == 202  # Check that the training session started
    assert mock_download_and_run.call_count == 1

    # While the download is in progress, check if we can still access other endpoints
    response = client.get('/training')
    assert response.status_code == 200  # Ensure we can still get training sessions

# @patch.object(TrainingSessionManager, 'run_training')
# def test_abort_training(client, session_manager, mock_run_training):
#     mock_run_training.return_value = None  # Prevent any real training from happening

#     # Start a training session
#     response = client.post('/start_training', json={
#         "pretrained_model_name_or_path": "path/to/model",
#         "train_data_dir": "path/to/train/data",
#         "output_dir": "path/to/output",
#         "resolution": 1024,
#         "train_batch_size": 1,
#         "learning_rate": 1e-5,
#         "max_train_steps": 1000,
#         "checkpointing_steps": 500,
#         "validation_prompt": "A sample validation prompt",
#         "num_validation_images": 4,
#         "validation_steps": 100,
#         "network_module": "networks.lora",
#         "network_dim": 32,
#         "network_alpha": 16,
#         "api_key": "test_api_key",
#         "checkpoint_url": "https://civitai.com/checkpoint"
#     })

#     assert response.status_code == 202
#     data = json.loads(response.data)
#     session_id = data['session_id']

#     # Abort the training session
#     response = client.delete(f'/abort_training/{session_id}')
#     assert response.status_code == 200
#     assert json.loads(response.data)['message'] == "Training session aborted"