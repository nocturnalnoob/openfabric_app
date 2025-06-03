import logging
import os
from typing import Dict
import uuid

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.stub import Stub
from core.memory import MemoryHandler
from core.llm import LLMHandler
from core.pipeline import Pipeline

# Global configurations and handlers
configurations: Dict[str, ConfigClass] = dict()
memory_handler: MemoryHandler = None
llm_handler: LLMHandler = None
pipeline: Pipeline = None

# Constants for Openfabric app IDs
TEXT2IMG_APP_ID = "f0997a01-d6d3-a5fe-53d8-561300318557"
IMG2_3D_APP_ID = "69543f29-4d41-4afc-7f29-3d51591f11eb"

############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data and initializes handlers.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application.
    """
    global memory_handler, llm_handler, pipeline
    
    # Initialize handlers if not already done
    if memory_handler is None:
        memory_handler = MemoryHandler(os.path.join("datastore", "memory.sqlite"))
        
    if llm_handler is None:
        # Note: Model path should be configured appropriately
        model_path = os.getenv("LLM_MODEL_PATH", "models/llama-2-7b.gguf")
        try:
            llm_handler = LLMHandler(model_path)
        except FileNotFoundError:
            logging.warning(f"LLM model not found at {model_path}. Some features may be limited.")
            
    if pipeline is None:
        pipeline = Pipeline(llm_handler, memory_handler, TEXT2IMG_APP_ID, IMG2_3D_APP_ID)
    
    # Save configurations
    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf


############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.
    Processes user prompts through the creative AI pipeline.

    Args:
        model (AppModel): The model object containing request and response structures.
    """
    # Retrieve input
    request: InputClass = model.request
    if not request.prompt:
        model.response.message = "Error: No prompt provided"
        return

    # Retrieve user config and initialize stub
    user_config: ConfigClass = configurations.get('super-user', None)
    if not user_config or not user_config.app_ids:
        model.response.message = "Error: No valid configuration found"
        return

    # Initialize the Stub with app IDs
    stub = Stub(user_config.app_ids)
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Process through pipeline
        result = pipeline.process(
            prompt=request.prompt,
            stub=stub,
            session_id=session_id
        )
        
        if result['status'] == 'success':
            # Prepare success response
            response_text = (
                f"Successfully processed your request!\n"
                f"Session ID: {session_id}\n"
                f"Enhanced prompt: {result['prompt']['expanded_prompt'][:200]}...\n"
                f"Generated assets:\n"
                f"- Image: {os.path.basename(result['image_path'])}\n"
                f"- 3D Model: {os.path.basename(result['model_path'])}"
            )
        else:
            response_text = f"Error in pipeline: {result.get('error', 'Unknown error')}"
            
        # Prepare response
        response: OutputClass = model.response
        response.message = response_text
        
    except Exception as e:
        logging.error(f"Pipeline execution error: {str(e)}")
        model.response.message = f"Error processing request: {str(e)}"