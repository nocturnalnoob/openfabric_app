from typing import Dict, Any
from llama_cpp import Llama
import os
import logging

class LLMHandler:
    """
    Handles interaction with local LLM models for prompt expansion and creative enhancement.
    Currently supports Llama models via llama.cpp.
    """
    def __init__(self, model_path: str):
        """
        Initialize LLM handler with a local model.
        
        Args:
            model_path (str): Path to the local LLM model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window
            n_threads=4   # CPU threads
        )
        
    def expand_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Expand and enhance a user prompt with creative details.
        
        Args:
            prompt (str): Original user prompt
            
        Returns:
            Dict containing:
            - expanded_prompt: Enhanced prompt with artistic details
            - style_hints: Specific style suggestions
            - technical_params: Any relevant technical parameters
        """
        system_context = """You are an expert artistic director specializing in both 2D and 3D art. 
        Enhance and expand the given prompt with vivid, specific details that will guide both image 
        generation and 3D modeling. Focus on lighting, composition, perspective, materials, and mood."""
        
        full_prompt = f"{system_context}\nOriginal prompt: {prompt}\nEnhanced description:"
        
        try:
            # Generate enhanced prompt
            response = self.llm(
                full_prompt,
                max_tokens=512,
                temperature=0.7,
                stop=["Original prompt:", "\n\n"]
            )
            
            # Extract main concepts for 3D conversion hints
            technical_prompt = "Extract key technical aspects for 3D modeling:"
            tech_response = self.llm(
                response['choices'][0]['text'] + "\n" + technical_prompt,
                max_tokens=256,
                temperature=0.3
            )
            
            return {
                'expanded_prompt': response['choices'][0]['text'].strip(),
                'style_hints': {
                    'lighting': self._extract_lighting(response['choices'][0]['text']),
                    'composition': self._extract_composition(response['choices'][0]['text'])
                },
                'technical_params': tech_response['choices'][0]['text'].strip()
            }
            
        except Exception as e:
            logging.error(f"LLM processing error: {str(e)}")
            return {
                'expanded_prompt': prompt,
                'style_hints': {},
                'technical_params': ""
            }
            
    def _extract_lighting(self, text: str) -> Dict[str, Any]:
        """Extract lighting information from expanded prompt."""
        # Basic extraction - could be enhanced with another LLM pass
        return {
            'primary_light': 'natural' if 'sunlight' in text.lower() else 'artificial',
            'mood': 'bright' if any(x in text.lower() for x in ['bright', 'sunny', 'daylight']) else 'dark'
        }
        
    def _extract_composition(self, text: str) -> Dict[str, Any]:
        """Extract composition hints from expanded prompt."""
        return {
            'perspective': 'wide' if any(x in text.lower() for x in ['panorama', 'wide', 'vast']) else 'close',
            'focus': 'foreground' if 'foreground' in text.lower() else 'background'
        }
