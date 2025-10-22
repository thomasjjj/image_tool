"""
OpenAI API response handler module
Provides utilities for interacting with OpenAI's chat completion API
"""

import os
import logging
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def respond(
    conversation: List[Dict[str, str]], 
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Send a conversation to OpenAI's chat completion API and return the response.
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys
        model: The model to use (default: gpt-4)
        temperature: Sampling temperature (0-2, default: 0.7)
        max_tokens: Maximum tokens in response (default: None for model default)
        api_key: Optional API key override (default: uses environment variable)
    
    Returns:
        The assistant's response as a string
        
    Raises:
        Exception: If the API call fails
    """
    try:
        # Use provided API key or get from environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("No API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Prepare the API call parameters
        params = {
            "model": model,
            "messages": conversation,
            "temperature": temperature
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Make the API call
        logger.debug(f"Calling OpenAI API with model: {model}")
        response = client.chat.completions.create(**params)
        
        # Extract and return the response
        assistant_message = response.choices[0].message.content
        logger.debug(f"Received response of length: {len(assistant_message)}")
        
        return assistant_message
        
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        raise


def respond_with_functions(
    conversation: List[Dict[str, str]],
    functions: List[Dict],
    model: str = "gpt-4",
    temperature: float = 0.7,
    api_key: Optional[str] = None
) -> Dict:
    """
    Send a conversation with function definitions to OpenAI's chat completion API.
    
    Args:
        conversation: List of message dictionaries
        functions: List of function definitions for the model to potentially call
        model: The model to use
        temperature: Sampling temperature
        api_key: Optional API key override
    
    Returns:
        Dictionary containing the full response including any function calls
    """
    try:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("No API key provided.")
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            functions=functions,
            temperature=temperature
        )
        
        return response.to_dict()
        
    except Exception as e:
        logger.error(f"Error in OpenAI API call with functions: {str(e)}")
        raise


def stream_response(
    conversation: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.7,
    api_key: Optional[str] = None
):
    """
    Stream a response from OpenAI's chat completion API.
    
    Args:
        conversation: List of message dictionaries
        model: The model to use
        temperature: Sampling temperature
        api_key: Optional API key override
    
    Yields:
        Chunks of the assistant's response as they arrive
    """
    try:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("No API key provided.")
        
        client = OpenAI(api_key=api_key)
        
        stream = client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=temperature,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Error in streaming OpenAI API call: {str(e)}")
        raise


# Model name mappings for compatibility
MODEL_MAPPINGS = {
    "gpt-5": "gpt-4",  # Map GPT-5 to GPT-4 since GPT-5 doesn't exist yet
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo-preview",
    "gpt-3.5": "gpt-3.5-turbo",
    "gpt-3": "gpt-3.5-turbo"
}


def get_model_name(requested_model: str) -> str:
    """
    Get the actual model name to use based on the requested model.
    
    Args:
        requested_model: The model name requested by the user
    
    Returns:
        The actual model name to use with the API
    """
    return MODEL_MAPPINGS.get(requested_model.lower(), requested_model)


# Override the default respond function to handle model mapping
_original_respond = respond

def respond(conversation: List[Dict[str, str]], model: str = "gpt-4", **kwargs) -> str:
    """
    Enhanced respond function with model name mapping.
    """
    actual_model = get_model_name(model)
    if actual_model != model:
        logger.info(f"Model '{model}' mapped to '{actual_model}'")
    return _original_respond(conversation, model=actual_model, **kwargs)
