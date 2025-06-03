from typing import Dict, Any, Optional
import logging
import tempfile
import os
from PIL import Image
import base64
import json

from core.memory import MemoryHandler
from core.llm import LLMHandler

class Pipeline:
    """
    Manages the end-to-end pipeline from prompt to 3D model, integrating:
    - LLM prompt expansion
    - Text-to-Image generation
    - Image-to-3D conversion
    - Memory management
    """
    def __init__(self, 
                llm_handler: LLMHandler,
                memory_handler: MemoryHandler,
                text2img_app_id: str,
                img2_3d_app_id: str):
        """
        Initialize the pipeline with required components.
        
        Args:
            llm_handler: Handler for local LLM interactions
            memory_handler: Handler for session and persistent memory
            text2img_app_id: Openfabric app ID for text-to-image service
            img2_3d_app_id: Openfabric app ID for image-to-3D service
        """
        self.llm = llm_handler
        self.memory = memory_handler
        self.text2img_id = text2img_app_id
        self.img2_3d_id = img2_3d_app_id
        
    async def process(self, 
                   prompt: str,
                   stub: Any,
                   session_id: str) -> Dict[str, Any]:
        """
        Process a prompt through the entire pipeline.
        
        Args:
            prompt: User's input prompt
            stub: Openfabric stub for API calls
            session_id: Unique session identifier
            
        Returns:
            Dict containing pipeline results and metadata
        """
        try:
            # 1. Expand prompt with LLM
            expanded = self.llm.expand_prompt(prompt)
            self.memory.save_session(f"prompt_{session_id}", {
                'original': prompt,
                'expanded': expanded
            })
            
            # 2. Generate image from expanded prompt
            img_result = stub.call(
                self.text2img_id,
                {'prompt': expanded['expanded_prompt']},
                session_id
            )
            
            img_path = self._save_temp_image(img_result['image'])
            self.memory.save_session(f"image_{session_id}", {
                'path': img_path,
                'metadata': expanded['style_hints']
            })
            
            # 3. Convert image to 3D
            model_result = stub.call(
                self.img2_3d_id,
                {
                    'image': img_path,
                    'params': expanded['technical_params']
                },
                session_id
            )
            
            model_path = self._save_temp_model(model_result['model'])
            self.memory.save_session(f"model_{session_id}", {
                'path': model_path,
                'metadata': model_result.get('metadata', {})
            })
            
            # 4. Save to persistent storage for long-term memory
            self.memory.save_persistent(f"creation_{session_id}", {
                'prompt': {
                    'original': prompt,
                    'expanded': expanded
                },
                'image': {
                    'path': img_path,
                    'metadata': expanded['style_hints']
                },
                'model': {
                    'path': model_path,
                    'metadata': model_result.get('metadata', {})
                }
            })
            
            return {
                'status': 'success',
                'prompt': expanded,
                'image_path': img_path,
                'model_path': model_path,
                'session_id': session_id
            }
            
        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'session_id': session_id
            }
            
    def _save_temp_image(self, image_data: bytes) -> str:
        """Save image data to a temporary file."""
        try:
            # Create a temporary file with .png extension
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                if isinstance(image_data, str):
                    # Handle base64 encoded strings
                    img_data = base64.b64decode(image_data)
                else:
                    img_data = image_data
                    
                tmp.write(img_data)
                return tmp.name
                
        except Exception as e:
            logging.error(f"Error saving image: {str(e)}")
            raise
            
    def _save_temp_model(self, model_data: bytes) -> str:
        """Save 3D model data to a temporary file."""
        try:
            # Create a temporary file with .glb extension
            with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
                if isinstance(model_data, str):
                    # Handle base64 encoded strings
                    model_data = base64.b64decode(model_data)
                    
                tmp.write(model_data)
                return tmp.name
                
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
