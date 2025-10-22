import streamlit as st
import os
import base64
import json
import asyncio
from datetime import datetime
from openai import AsyncOpenAI, OpenAI
from typing import List, Dict, Any, Optional
import re
import time

# Configure Streamlit page
st.set_page_config(
    page_title="AI Image Generator Chat",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS for terminal-like appearance
st.markdown("""
<style>
.terminal-text {
    font-family: 'Courier New', monospace;
    background-color: #1e1e1e;
    color: #00ff00;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.command-text {
    color: #ffff00;
}
.system-message {
    color: #00ffff;
}
.error-message {
    color: #ff0000;
}
</style>
""", unsafe_allow_html=True)


def terminal_print(message: str, message_type: str = "info"):
    """Display a terminal-like message in the chat."""
    color_class = {
        "info": "system-message",
        "command": "command-text",
        "error": "error-message",
        "success": "system-message"
    }.get(message_type, "system-message")
    
    return f'<div class="terminal-text"><span class="{color_class}">{message}</span></div>'


def respond(conversation: List[Dict], model: str = "gpt-4", temperature: float = 0.7) -> str:
    """
    Simple implementation of the respond function using OpenAI API.
    Replace this with your actual utils.openai_responses.respond if needed.
    """
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        response = client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return ""


def save_generation_metadata(folder: str, data: Dict[str, Any]) -> None:
    """Save metadata about the generation run to a text file."""
    metadata_file = os.path.join(folder, "generation_info.txt")
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("IMAGE GENERATION METADATA\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if 'original_prompt' in data:
            f.write("ORIGINAL PROMPT:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{data['original_prompt']}\n\n")
        
        if 'clarifying_questions' in data and data['clarifying_questions']:
            f.write("CLARIFYING QUESTIONS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{data['clarifying_questions']}\n\n")
        
        if 'user_answers' in data and data['user_answers']:
            f.write("USER ANSWERS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{data['user_answers']}\n\n")
        
        if 'generated_prompts' in data:
            f.write("GENERATED PROMPTS:\n")
            f.write("-" * 70 + "\n")
            for i, prompt in enumerate(data['generated_prompts'], 1):
                f.write(f"{i}. {prompt}\n\n")
        
        f.write("=" * 70 + "\n")
    
    return metadata_file


async def generate_image_async(
    prompt: str, 
    index: int, 
    total: int, 
    run_folder: str,
    api_key: str,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard"
) -> Dict[str, Any]:
    """Generate a single image using OpenAI's DALL-E API asynchronously."""
    os.makedirs(run_folder, exist_ok=True)
    
    try:
        st.session_state.messages.append({
            "role": "system",
            "content": terminal_print(f"[{index}/{total}] Starting: {prompt[:60]}...", "command")
        })
        
        # Create async client with the API key
        async_client = AsyncOpenAI(api_key=api_key)
        
        # Make async API request
        response = await async_client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,
            quality=quality,
            response_format="b64_json"
        )
        
        # Decode and save image
        image_bytes = base64.b64decode(response.data[0].b64_json)
        
        # Create safe filename
        safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt.replace(' ', '_')
        filename = f"{index:02d}_{safe_prompt}.png"
        filepath = os.path.join(run_folder, filename)
        
        # Save image
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        
        st.session_state.messages.append({
            "role": "system",
            "content": terminal_print(f"[{index}/{total}] ✅ Completed: {filepath}", "success")
        })
        
        return {
            'success': True,
            'filepath': filepath,
            'prompt': prompt,
            'image_bytes': image_bytes
        }
        
    except Exception as e:
        error_msg = f"[{index}/{total}] ❌ Error: {str(e)}"
        st.session_state.messages.append({
            "role": "system",
            "content": terminal_print(error_msg, "error")
        })
        return {
            'success': False,
            'filepath': None,
            'prompt': prompt,
            'error': str(e)
        }


async def generate_images_concurrent(prompts: List[str], run_folder: str, api_key: str) -> List[Dict[str, Any]]:
    """Generate multiple images concurrently using asyncio."""
    total = len(prompts)
    
    tasks = [
        generate_image_async(prompt, i + 1, total, run_folder, api_key)
        for i, prompt in enumerate(prompts)
    ]
    
    results = await asyncio.gather(*tasks)
    return results


def improve_prompt_with_gpt(
    original_prompt: str, 
    num_variations: int,
    clarifying_questions: Optional[str] = None,
    user_answers: Optional[str] = None
) -> List[str]:
    """Use GPT to generate improved prompt variations."""
    
    st.session_state.messages.append({
        "role": "system",
        "content": terminal_print("🤖 GPT is generating improved prompt variations...", "info")
    })
    
    variation_prompt = f"""Based on the original prompt: '{original_prompt}'
{f"And the user's additional input: {user_answers}" if user_answers else ""}

Generate {num_variations} improved, detailed image generation prompts.

Each prompt should:
- Be specific and detailed
- Include style, mood, lighting, and composition details
- Be optimized for high-quality image generation
- Offer meaningful variations (different angles, styles, or interpretations)

Return ONLY a JSON array of strings, nothing else. Format:
["prompt 1 here", "prompt 2 here", "prompt 3 here"]

DO NOT include any markdown formatting, explanations, or text outside the JSON array."""
    
    conversation = [
        {"role": "system", "content": "You are an expert at creating detailed image generation prompts. Return only valid JSON."},
        {"role": "user", "content": variation_prompt}
    ]
    
    response = respond(conversation, model="gpt-4", temperature=0.8)
    
    try:
        # Clean the response
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()
        
        variations = json.loads(clean_response)
        
        if isinstance(variations, list) and len(variations) > 0:
            return variations[:num_variations]
        else:
            return [original_prompt]
            
    except json.JSONDecodeError:
        st.error("Failed to parse GPT response. Using original prompt.")
        return [original_prompt]


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = None
    if "current_prompts" not in st.session_state:
        st.session_state.current_prompts = []
    if "generation_state" not in st.session_state:
        st.session_state.generation_state = "idle"
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    
    # Title
    st.title("🎨 AI Image Generator Chat Interface")
    st.markdown("*Terminal-style chat interface for AI image generation*")
    
    # API Key Input (Always shown at the top)
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input(
                "🔑 Enter your OpenAI API Key:",
                type="password",
                placeholder="sk-...",
                help="Your API key is not stored and must be entered each session"
            )
        with col2:
            if st.button("Set API Key", type="primary"):
                if api_key:
                    st.session_state.api_key = api_key
                    st.session_state.messages.append({
                        "role": "system",
                        "content": terminal_print("✅ API key set successfully!", "success")
                    })
                    st.success("API key set!")
                else:
                    st.error("Please enter an API key")
    
    # Only show the rest of the interface if API key is set
    if not st.session_state.api_key:
        st.warning("⚠️ Please enter your OpenAI API key to continue")
        return
    
    # Create two columns for chat and images
    chat_col, image_col = st.columns([2, 1])
    
    with chat_col:
        st.subheader("💬 Chat Terminal")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                elif message["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                elif message["role"] == "system":
                    st.markdown(message["content"], unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Enter your image prompt or command..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process the command
            if prompt.lower() in ['quit', 'exit', 'q']:
                st.session_state.messages.append({
                    "role": "system",
                    "content": terminal_print("👋 Goodbye! Refresh the page to start a new session.", "info")
                })
                
            elif prompt.lower() == 'help':
                help_text = """
Available commands:
- Enter any image prompt to generate images
- 'improve' - Use GPT to improve your last prompt
- 'regenerate' - Generate more images with the same prompts
- 'adjust' - Adjust the current prompts
- 'clear' - Clear the chat history
- 'quit' / 'exit' - End the session
                """
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": help_text
                })
                
            elif prompt.lower() == 'clear':
                st.session_state.messages = []
                st.session_state.generated_images = []
                st.rerun()
                
            elif prompt.lower() == 'improve' and st.session_state.current_prompts:
                # Improve existing prompts
                with st.spinner("Improving prompts..."):
                    improved_prompts = improve_prompt_with_gpt(
                        st.session_state.current_prompts[0] if st.session_state.current_prompts else "",
                        3
                    )
                    st.session_state.current_prompts = improved_prompts
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "✅ Improved prompts:\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(improved_prompts)])
                    })
                    
            elif prompt.lower() == 'regenerate' and st.session_state.current_prompts:
                # Regenerate with current prompts
                run_generation = True
                prompts_to_generate = st.session_state.current_prompts
                
            else:
                # New prompt - ask about improvements
                st.session_state.current_prompts = [prompt]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Would you like me to improve this prompt with GPT? Type 'yes' for improvements or 'no' to generate as-is."
                })
                
                # Wait for user response (this will be handled in the next interaction)
                st.session_state.generation_state = "awaiting_improve_decision"
        
        # Handle generation state
        if st.session_state.generation_state == "awaiting_improve_decision":
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                last_response = st.session_state.messages[-1]["content"].lower()
                
                if last_response in ['yes', 'y']:
                    with st.spinner("Improving prompts..."):
                        improved_prompts = improve_prompt_with_gpt(
                            st.session_state.current_prompts[0],
                            3
                        )
                        st.session_state.current_prompts = improved_prompts
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "✅ Generated improved prompts:\n" + "\n".join([f"{i+1}. {p}" for i, p in enumerate(improved_prompts)])
                        })
                        run_generation = True
                        
                elif last_response in ['no', 'n']:
                    run_generation = True
                else:
                    run_generation = False
                    
                st.session_state.generation_state = "idle"
                
                # Generate images if confirmed
                if 'run_generation' in locals() and run_generation:
                    with st.spinner(f"Generating {len(st.session_state.current_prompts)} image(s)..."):
                        # Create folder for this generation
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        run_folder = os.path.join("generated_images", timestamp)
                        os.makedirs(run_folder, exist_ok=True)
                        
                        # Save metadata
                        metadata = {
                            'original_prompt': st.session_state.current_prompts[0] if st.session_state.current_prompts else "",
                            'generated_prompts': st.session_state.current_prompts
                        }
                        metadata_file = save_generation_metadata(run_folder, metadata)
                        
                        st.session_state.messages.append({
                            "role": "system",
                            "content": terminal_print(f"📁 Session folder: {run_folder}", "info")
                        })
                        
                        # Run async generation
                        results = asyncio.run(generate_images_concurrent(
                            st.session_state.current_prompts,
                            run_folder,
                            st.session_state.api_key
                        ))
                        
                        # Process results
                        successful = sum(1 for r in results if r['success'])
                        failed = len(results) - successful
                        
                        summary = f"""
✨ Generation Complete!
✅ Successful: {successful}
{f'❌ Failed: {failed}' if failed > 0 else ''}
📝 Metadata saved to: {metadata_file}
                        """
                        st.session_state.messages.append({
                            "role": "system",
                            "content": terminal_print(summary, "success")
                        })
                        
                        # Store generated images
                        for result in results:
                            if result['success'] and 'image_bytes' in result:
                                st.session_state.generated_images.append({
                                    'prompt': result['prompt'],
                                    'filepath': result['filepath'],
                                    'image_bytes': result['image_bytes']
                                })
                        
                        st.rerun()
    
    # Image display column
    with image_col:
        st.subheader("🖼️ Generated Images")
        
        if st.session_state.generated_images:
            for idx, img_data in enumerate(st.session_state.generated_images[-5:], 1):  # Show last 5 images
                with st.expander(f"Image {idx}: {img_data['prompt'][:30]}...", expanded=True):
                    st.image(img_data['image_bytes'], use_column_width=True)
                    st.caption(f"Prompt: {img_data['prompt']}")
                    st.text(f"File: {img_data['filepath']}")
                    
                    # Download button
                    st.download_button(
                        label="Download",
                        data=img_data['image_bytes'],
                        file_name=os.path.basename(img_data['filepath']),
                        mime="image/png"
                    )
        else:
            st.info("Generated images will appear here...")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("📖 Instructions")
        st.markdown("""
        1. **Enter your OpenAI API key** at the top
        2. **Type an image prompt** in the chat
        3. **Choose to improve** the prompt with GPT or generate as-is
        4. **View generated images** in the right panel
        
        ### Commands:
        - `help` - Show available commands
        - `improve` - Improve current prompts
        - `regenerate` - Generate more with same prompts
        - `adjust` - Modify current prompts
        - `clear` - Clear chat history
        - `quit` / `exit` - End session
        
        ### Notes:
        - Images are saved to `generated_images/` folder
        - Metadata is saved with each generation
        - API key is required for each session
        """)


if __name__ == "__main__":
    main()
