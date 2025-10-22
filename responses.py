"""
OpenAI API response handler module
Provides utilities for interacting with OpenAI's Responses API
"""

import os
import logging
from typing import List, Dict, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_conversation_for_responses(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert simple role/content pairs into Responses API message format."""
    formatted: List[Dict[str, Any]] = []
    for message in conversation:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            formatted.append({"role": role, "content": content})
        else:
            formatted.append({
                "role": role,
                "content": [{"type": "text", "text": str(content)}]
            })
    return formatted


def _extract_text_from_response(response: Any) -> str:
    """Safely extract assistant text from a Responses API payload."""
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    text_parts: List[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "text":
                    text_parts.append(getattr(content, "text", ""))
    return "".join(text_parts)


def respond(
    conversation: List[Dict[str, Any]],
    model: str = "gpt-4.1",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None
) -> str:
    """Send a conversation to the OpenAI Responses API and return the assistant text."""
    try:
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("No API key provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        client = OpenAI(api_key=api_key)

        params: Dict[str, Any] = {
            "model": model,
            "input": _format_conversation_for_responses(conversation),
            "temperature": temperature,
        }

        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        logger.debug(f"Calling OpenAI Responses API with model: {model}")
        response = client.responses.create(**params)

        assistant_message = _extract_text_from_response(response)
        logger.debug(f"Received response of length: {len(assistant_message)}")

        return assistant_message

    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        raise


def respond_with_functions(
    conversation: List[Dict[str, Any]],
    functions: List[Dict],
    model: str = "gpt-4.1",
    temperature: float = 0.7,
    api_key: Optional[str] = None
) -> Dict:
    """Send a conversation with tool definitions to the Responses API."""
    try:
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("No API key provided.")

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=model,
            input=_format_conversation_for_responses(conversation),
            tools=functions,
            temperature=temperature,
        )

        return response.to_dict()

    except Exception as e:
        logger.error(f"Error in OpenAI API call with functions: {str(e)}")
        raise


def stream_response(
    conversation: List[Dict[str, Any]],
    model: str = "gpt-4.1",
    temperature: float = 0.7,
    api_key: Optional[str] = None
):
    """Stream a response from the OpenAI Responses API."""
    try:
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("No API key provided.")

        client = OpenAI(api_key=api_key)

        stream = client.responses.stream(
            model=model,
            input=_format_conversation_for_responses(conversation),
            temperature=temperature,
        )

        for event in stream:
            if getattr(event, "type", None) == "response.output_text.delta":
                yield getattr(event, "delta", "")
            elif getattr(event, "type", None) == "response.completed":
                break

    except Exception as e:
        logger.error(f"Error in streaming OpenAI API call: {str(e)}")
        raise


# Model name mappings for compatibility
MODEL_MAPPINGS = {
    "gpt-5": "gpt-4.1",  # Map legacy aliases to current flagship models
    "gpt-4": "gpt-4.1",
    "gpt-4-turbo": "gpt-4.1-mini",
    "gpt-3.5": "gpt-4.1-mini",
    "gpt-3": "gpt-4.1-mini",
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

def respond(conversation: List[Dict[str, Any]], model: str = "gpt-4.1", **kwargs) -> str:
    """
    Enhanced respond function with model name mapping.
    """
    actual_model = get_model_name(model)
    if actual_model != model:
        logger.info(f"Model '{model}' mapped to '{actual_model}'")
    return _original_respond(conversation, model=actual_model, **kwargs)
