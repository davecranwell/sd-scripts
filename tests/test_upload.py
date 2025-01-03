from training_session_manager import TrainingSessionManager
import os

# Create an instance of TrainingSessionManager
manager = TrainingSessionManager()

# Use the function

file_path = os.path.join('/home/dave/stable-diffusion-webui-docker/data/models/Lora', '')
presigned_url = ""  # your presigned URL

# Create a mock session ID for testing
test_session_id = "test_session_123"

create_training_session_response = manager.create_training_session({
    "id": test_session_id,
})

try:
    # Use the _upload_file method directly
    manager._upload_file(
        file_path=file_path,
        url=presigned_url,
        session_id=test_session_id,
        status_prefix="uploading"
    )
except Exception as e:
    print(f"Upload failed: {str(e)}")