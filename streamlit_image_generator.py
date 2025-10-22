import streamlit as st
import os
import base64
import json
import asyncio
from datetime import datetime
from openai import AsyncOpenAI, OpenAI
from typing import List, Dict, Any
import re
import time
import io
import zipfile

# Configure Streamlit page
st.set_page_config(
    page_title="AI Image Generator Chat",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for terminal-like appearance and better UI
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
.command-text { color: #ffff00; }
.system-message { color: #00ffff; }
.error-message { color: #ff0000; }
.success-message { color: #00ff00; }
.stButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


class ImageGeneratorChat:
    """Main class for the Image Generator Chat Application"""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            "messages": [],
            "api_key": None,
            "current_prompts": [],
            "original_prompt": "",
            "generation_state": "idle",
            "generated_images": [],
            "clarifying_questions": "",
            "user_answers": "",
            "awaiting_response": None,
            "generation_counter": 0,
            "session_folder": None,
            "russian_guardrail": False,
            "num_variations": 3
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def terminal_print(self, message: str, message_type: str = "info") -> str:
        """Display a terminal-like message"""
        color_map = {
            "info": "system-message",
            "command": "command-text",
            "error": "error-message",
            "success": "success-message"
        }
        color_class = color_map.get(message_type, "system-message")
        return f'<div class="terminal-text"><span class="{color_class}">{message}</span></div>'
    
    def add_message(self, role: str, content: str, message_type: str = "info"):
        """Add a message to the chat history"""
        if role == "system":
            content = self.terminal_print(content, message_type)
        st.session_state.messages.append({"role": role, "content": content})
    
    def _format_conversation_for_responses(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert legacy chat messages into the Responses API format."""
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

    def _extract_text_from_response(self, response: Any) -> str:
        """Safely extract text content from a Responses API response."""
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        text_chunks: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for piece in getattr(item, "content", []) or []:
                    if getattr(piece, "type", None) == "text":
                        text_chunks.append(getattr(piece, "text", ""))

        return "".join(text_chunks)

    def respond(self, conversation: List[Dict], model: str = "gpt-4.1", temperature: float = 0.7) -> str:
        """Call OpenAI Responses API for conversational completions."""
        try:
            client = OpenAI(api_key=st.session_state.api_key)
            formatted_input = self._format_conversation_for_responses(conversation)
            response = client.responses.create(
                model=model,
                input=formatted_input,
                temperature=temperature
            )
            return self._extract_text_from_response(response)
        except Exception as e:
            self.add_message("system", f"Error calling OpenAI API: {str(e)}", "error")
            return ""
    
    def save_generation_metadata(self, folder: str, data: Dict[str, Any]) -> str:
        """Save metadata about the generation run"""
        metadata_file = os.path.join(folder, "generation_info.txt")
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("IMAGE GENERATION METADATA\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, label in [
                ('original_prompt', 'ORIGINAL PROMPT'),
                ('clarifying_questions', 'CLARIFYING QUESTIONS (from GPT)'),
                ('user_answers', 'USER ANSWERS'),
                ('adjustment_request', 'ADJUSTMENT REQUEST')
            ]:
                if key in data and data[key]:
                    f.write(f"{label}:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"{data[key]}\n\n")
            
            if 'generated_prompts' in data:
                f.write("GENERATED PROMPTS:\n")
                f.write("-" * 70 + "\n")
                for i, prompt in enumerate(data['generated_prompts'], 1):
                    f.write(f"{i}. {prompt}\n\n")
            
            f.write("=" * 70 + "\n")
        
        return metadata_file
    
    async def generate_image_async(self, prompt: str, index: int, total: int, 
                                  run_folder: str, api_key: str) -> Dict[str, Any]:
        """Generate a single image asynchronously"""
        os.makedirs(run_folder, exist_ok=True)
        
        try:
            self.add_message("system", f"[{index}/{total}] Generating: {prompt[:60]}...", "command")
            
            async_client = AsyncOpenAI(api_key=api_key)
            
            # Generate image using the Responses API image generation tool
            response = await async_client.responses.create(
                model="gpt-4.1",
                input=prompt,
                tools=[
                    {
                        "type": "image_generation",
                        "size": "1024x1024",
                        "quality": "high",
                    }
                ],
            )

            image_base64 = None
            for output in getattr(response, "output", []) or []:
                if getattr(output, "type", None) == "image_generation_call":
                    image_base64 = getattr(output, "result", None)
                    break

            if not image_base64:
                raise ValueError("No image data returned from the API")

            # Decode and save image
            image_bytes = base64.b64decode(image_base64)
            
            # Create safe filename
            safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')
            filename = f"{index:02d}_{safe_prompt}.png"
            filepath = os.path.join(run_folder, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_bytes)
            
            self.add_message("system", f"[{index}/{total}] ✅ Saved to: {filepath}", "success")
            
            return {
                'success': True,
                'filepath': filepath,
                'prompt': prompt,
                'image_bytes': image_bytes
            }
            
        except Exception as e:
            self.add_message("system", f"[{index}/{total}] ❌ Error: {str(e)}", "error")
            return {
                'success': False,
                'filepath': None,
                'prompt': prompt,
                'error': str(e)
            }
    
    async def generate_images_concurrent(self, prompts: List[str], run_folder: str) -> List[Dict[str, Any]]:
        """Generate multiple images concurrently"""
        total = len(prompts)
        api_key = st.session_state.api_key
        
        tasks = [
            self.generate_image_async(prompt, i + 1, total, run_folder, api_key)
            for i, prompt in enumerate(prompts)
        ]
        
        results = await asyncio.gather(*tasks)
        return results
    
    def get_clarifying_questions(self, prompt: str) -> str:
        """Get clarifying questions from GPT about the prompt"""
        russian_instruction = ""
        if st.session_state.russian_guardrail:
            russian_instruction = "\n\nIMPORTANT: Any text in the image MUST be in Russian (Cyrillic script)."
        
        system_prompt = f"""You are an expert at creating detailed image generation prompts. 
Your job is to help users create better prompts by asking 1-3 thoughtful clarifying questions.

Ask about:
- Style preferences (photorealistic, artistic, cartoon, etc.)
- Mood or atmosphere
- Important details that might be missing
- Color preferences
- Composition or framing
{russian_instruction}

Keep questions concise and focused. Ask only what's necessary to significantly improve the prompt."""
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"I want to generate an image with this prompt: '{prompt}'\n\nWhat questions do you have to help me create better image prompts?"}
        ]
        
        return self.respond(conversation, model="gpt-4.1", temperature=0.7)
    
    def improve_prompts_with_gpt(self, original_prompt: str, num_variations: int, 
                                answers: str = "") -> List[str]:
        """Generate improved prompt variations using GPT"""
        russian_requirement = ""
        if st.session_state.russian_guardrail:
            russian_requirement = "\n\nCRITICAL: Any text visible in the image MUST be in Russian (Cyrillic script)."
        
        variation_prompt = f"""Based on the original prompt: '{original_prompt}'
{f"And the user's additional input: {answers}" if answers else ""}

Generate {num_variations} improved, detailed image generation prompts.

Each prompt should:
- Be specific and detailed
- Include style, mood, lighting, and composition details
- Be optimized for high-quality image generation
- Offer meaningful variations (different angles, styles, or interpretations)
{russian_requirement}

Return ONLY a JSON array of strings, nothing else. Format:
["prompt 1 here", "prompt 2 here", "prompt 3 here"]"""
        
        conversation = [
            {"role": "system", "content": "You are an expert at creating detailed image generation prompts. Return only valid JSON."},
            {"role": "user", "content": variation_prompt}
        ]
        
        response = self.respond(conversation, model="gpt-4.1", temperature=0.8)
        
        try:
            # Clean response
            clean_response = response.strip()
            for prefix in ["```json", "```"]:
                if clean_response.startswith(prefix):
                    clean_response = clean_response[len(prefix):]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            variations = json.loads(clean_response)
            if isinstance(variations, list) and len(variations) > 0:
                return variations[:num_variations]
        except json.JSONDecodeError:
            self.add_message("system", "Failed to parse GPT response. Using original prompt.", "error")
        
        return [original_prompt]
    
    def adjust_prompts(self, current_prompts: List[str], adjustment: str) -> List[str]:
        """Adjust existing prompts based on user feedback"""
        russian_requirement = ""
        if st.session_state.russian_guardrail:
            russian_requirement = "\n\nCRITICAL: Maintain the requirement that ALL text in images must be in Russian."
        
        adjustment_prompt = f"""I have these image generation prompts:
{chr(10).join(f'{i + 1}. {p}' for i, p in enumerate(current_prompts))}

The user wants to make this adjustment: {adjustment}

Generate {len(current_prompts)} ADJUSTED prompts that incorporate the user's feedback.
{russian_requirement}

Return ONLY a JSON array of strings, nothing else."""
        
        conversation = [
            {"role": "system", "content": "You are an expert at refining image generation prompts. Return only valid JSON."},
            {"role": "user", "content": adjustment_prompt}
        ]
        
        response = self.respond(conversation, model="gpt-4.1", temperature=0.7)
        
        try:
            clean_response = response.strip()
            for prefix in ["```json", "```"]:
                if clean_response.startswith(prefix):
                    clean_response = clean_response[len(prefix):]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            
            adjusted_prompts = json.loads(clean_response.strip())
            if isinstance(adjusted_prompts, list) and len(adjusted_prompts) > 0:
                return adjusted_prompts[:len(current_prompts)]
        except json.JSONDecodeError:
            self.add_message("system", "Failed to parse adjustment. Keeping current prompts.", "error")
        
        return current_prompts
    
    def process_command(self, command: str):
        """Process user commands and prompts"""
        cmd = command.lower().strip()
        
        # Handle special commands
        if cmd in ['quit', 'exit', 'q']:
            self.add_message("system", "👋 Goodbye! Refresh the page to start a new session.", "info")
            st.session_state.generation_state = "ended"
            
        elif cmd == 'help':
            help_text = """
**Available Commands:**
• **Any text** - Generate an image from your prompt
• **improve** - Improve current prompts with GPT
• **regenerate** - Generate more images with same prompts  
• **adjust** - Modify current prompts
• **variations [n]** - Set number of variations (1-10)
• **russian on/off** - Toggle Russian text requirement
• **clear** - Clear chat history
• **help** - Show this help message
• **quit/exit** - End session
            """
            st.session_state.messages.append({"role": "assistant", "content": help_text})
            
        elif cmd == 'clear':
            st.session_state.messages = []
            st.session_state.generated_images = []
            self.add_message("system", "✨ Chat cleared!", "success")
            
        elif cmd.startswith('variations'):
            try:
                num = int(cmd.split()[1])
                st.session_state.num_variations = max(1, min(10, num))
                self.add_message("system", f"✅ Set to generate {st.session_state.num_variations} variations", "success")
            except:
                self.add_message("system", "Usage: variations [1-10]", "error")
                
        elif cmd.startswith('russian'):
            if 'on' in cmd:
                st.session_state.russian_guardrail = True
                self.add_message("system", "🇷🇺 Russian text requirement ENABLED", "success")
            elif 'off' in cmd:
                st.session_state.russian_guardrail = False
                self.add_message("system", "🌍 Russian text requirement DISABLED", "success")
            else:
                status = "ON" if st.session_state.russian_guardrail else "OFF"
                self.add_message("system", f"Russian text requirement is {status}", "info")
                
        elif cmd == 'improve' and st.session_state.current_prompts:
            st.session_state.generation_state = "improving"
            
        elif cmd == 'regenerate' and st.session_state.current_prompts:
            st.session_state.generation_state = "regenerating"
            
        elif cmd == 'adjust' and st.session_state.current_prompts:
            st.session_state.generation_state = "awaiting_adjustment"
            st.session_state.messages.append({
                "role": "assistant",
                "content": "What adjustments would you like to make to the prompts?"
            })
            
        else:
            # New image prompt
            st.session_state.original_prompt = command
            st.session_state.current_prompts = [command]
            st.session_state.generation_state = "new_prompt"
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Would you like me to improve this prompt with GPT? (yes/no)"
            })
            st.session_state.awaiting_response = "improve_decision"
    
    def run_generation(self, prompts: List[str]):
        """Run the image generation process"""
        # Create session folder if needed
        if not st.session_state.session_folder:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in st.session_state.original_prompt[:30] 
                               if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
            st.session_state.session_folder = os.path.join("generated_images", f"{timestamp}_{safe_name}")
            os.makedirs(st.session_state.session_folder, exist_ok=True)
        
        # Create run folder
        st.session_state.generation_counter += 1
        if st.session_state.generation_counter == 1:
            run_folder = os.path.join(st.session_state.session_folder, "initial")
        else:
            run_folder = os.path.join(st.session_state.session_folder,
                                     f"run_{st.session_state.generation_counter:02d}")

        # Ensure the run folder exists before attempting to save any files
        os.makedirs(run_folder, exist_ok=True)

        # Save metadata
        metadata = {
            'original_prompt': st.session_state.original_prompt,
            'generated_prompts': prompts,
            'clarifying_questions': st.session_state.clarifying_questions,
            'user_answers': st.session_state.user_answers
        }
        metadata_file = self.save_generation_metadata(run_folder, metadata)
        
        self.add_message("system", f"📁 Saving to: {run_folder}", "info")
        self.add_message("system", f"🖼️ Generating {len(prompts)} image(s)...", "info")
        
        # Run async generation
        with st.spinner(f"Generating {len(prompts)} image(s)..."):
            results = asyncio.run(self.generate_images_concurrent(prompts, run_folder))
        
        # Process results
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        self.add_message("system", "=" * 70, "info")
        self.add_message("system", f"✨ Generation Complete!", "success")
        self.add_message("system", f"✅ Successful: {successful}", "success")
        if failed > 0:
            self.add_message("system", f"❌ Failed: {failed}", "error")
        self.add_message("system", f"📝 Metadata: {metadata_file}", "info")
        self.add_message("system", "=" * 70, "info")
        
        # Store generated images
        successful_results = []
        for result in results:
            if result['success'] and 'image_bytes' in result:
                image_entry = {
                    'prompt': result['prompt'],
                    'filepath': result['filepath'],
                    'image_bytes': result['image_bytes']
                }
                st.session_state.generated_images.append(image_entry)
                successful_results.append(image_entry)

        if successful_results:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for image in successful_results:
                    zipf.writestr(os.path.basename(image['filepath']), image['image_bytes'])
            zip_buffer.seek(0)

            gallery_id = f"run_{st.session_state.generation_counter:02d}_{int(time.time()*1000)}"
            st.session_state.messages.append({
                "role": "assistant",
                "type": "image_gallery",
                "id": gallery_id,
                "title": f"🎉 Generated {len(successful_results)} image{'s' if len(successful_results) != 1 else ''}",
                "images": [
                    {
                        "prompt": image['prompt'],
                        "image_bytes": image['image_bytes'],
                        "filename": os.path.basename(image['filepath'])
                    }
                    for image in successful_results
                ],
                "zip_data": zip_buffer.getvalue(),
                "zip_name": f"{os.path.basename(run_folder)}.zip"
            })
        
        # Offer next actions
        st.session_state.messages.append({
            "role": "assistant",
            "content": """
**What would you like to do next?**
• Type **regenerate** to generate more with same prompts
• Type **adjust** to modify the prompts
• Type **improve** to enhance prompts with GPT
• Enter a **new prompt** to start fresh
            """
        })
        
        st.session_state.generation_state = "idle"
    
    def render_ui(self):
        """Render the main UI"""
        st.title("🎨 AI Image Generator Chat Interface")
        st.markdown("*Terminal-style chat interface with GPT-powered prompt enhancement*")
        
        # API Key Input Section
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
                st.write("")  # Spacing
                if st.button("Set API Key", type="primary", use_container_width=True):
                    if api_key:
                        st.session_state.api_key = api_key
                        self.add_message("system", "✅ API key set successfully!", "success")
                        st.rerun()
                    else:
                        st.error("Please enter an API key")
        
        # Check for API key
        if not st.session_state.api_key:
            st.warning("⚠️ Please enter your OpenAI API key to continue")
            return
        
        # Main layout
        chat_col, image_col = st.columns([2, 1])
        
        with chat_col:
            st.subheader("💬 Chat Terminal")
            
            # Display messages
            message_container = st.container(height=500)
            with message_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.write(message["content"])
                    elif message["role"] == "assistant":
                        if message.get("type") == "image_gallery":
                            with st.chat_message("assistant"):
                                st.markdown(message.get("title", "**Generated Images**"))
                                for idx, image in enumerate(message.get("images", []), 1):
                                    st.image(image["image_bytes"], use_column_width=True)
                                    st.caption(f"**Prompt:** {image['prompt']}")
                                    st.download_button(
                                        label="📥 Download",
                                        data=image["image_bytes"],
                                        file_name=image["filename"],
                                        mime="image/png",
                                        key=f"download_{message['id']}_{idx}"
                                    )

                                if message.get("zip_data"):
                                    st.download_button(
                                        label="📦 Download All",
                                        data=message["zip_data"],
                                        file_name=message.get("zip_name", "generated_images.zip"),
                                        mime="application/zip",
                                        key=f"download_all_{message['id']}"
                                    )
                        else:
                            with st.chat_message("assistant"):
                                st.markdown(message["content"])
                    elif message["role"] == "system":
                        st.markdown(message["content"], unsafe_allow_html=True)
            
            # Chat input
            if user_input := st.chat_input("Enter command or image prompt..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Handle awaiting responses
                if st.session_state.awaiting_response == "improve_decision":
                    if user_input.lower() in ['yes', 'y']:
                        # Get clarifying questions
                        with st.spinner("Getting clarifying questions..."):
                            questions = self.get_clarifying_questions(st.session_state.original_prompt)
                            if questions:
                                st.session_state.clarifying_questions = questions
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"**GPT Questions:**\n{questions}\n\nPlease answer the questions (or press Enter to skip):"
                                })
                                st.session_state.awaiting_response = "clarifying_answers"
                            else:
                                # No questions, go straight to generation
                                st.session_state.awaiting_response = None
                                st.session_state.generation_state = "improving"
                    else:
                        # Generate without improvement
                        st.session_state.awaiting_response = None
                        self.run_generation(st.session_state.current_prompts)
                        
                elif st.session_state.awaiting_response == "clarifying_answers":
                    st.session_state.user_answers = user_input
                    st.session_state.awaiting_response = None
                    st.session_state.generation_state = "improving"
                    
                elif st.session_state.generation_state == "awaiting_adjustment":
                    # Process adjustment
                    with st.spinner("Adjusting prompts..."):
                        adjusted = self.adjust_prompts(st.session_state.current_prompts, user_input)
                        st.session_state.current_prompts = adjusted
                        
                        self.add_message("assistant", "✅ Adjusted prompts:")
                        for i, p in enumerate(adjusted, 1):
                            self.add_message("assistant", f"{i}. {p}")
                        
                        self.run_generation(adjusted)
                else:
                    # Process normal command
                    self.process_command(user_input)
                
                # Handle state-based actions
                if st.session_state.generation_state == "improving":
                    with st.spinner("Improving prompts with GPT..."):
                        improved = self.improve_prompts_with_gpt(
                            st.session_state.original_prompt,
                            st.session_state.num_variations,
                            st.session_state.user_answers
                        )
                        st.session_state.current_prompts = improved
                        
                        self.add_message("assistant", "✅ Generated improved prompts:")
                        for i, p in enumerate(improved, 1):
                            self.add_message("assistant", f"{i}. {p}")
                        
                        self.run_generation(improved)
                        
                elif st.session_state.generation_state == "regenerating":
                    self.run_generation(st.session_state.current_prompts)
                
                st.rerun()
        
        # Image display column
        with image_col:
            st.subheader("🖼️ Generated Images")
            
            if st.session_state.generated_images:
                # Show last 5 images
                for idx, img_data in enumerate(reversed(st.session_state.generated_images[-5:]), 1):
                    with st.expander(f"Image: {img_data['prompt'][:30]}...", expanded=(idx==1)):
                        st.image(img_data['image_bytes'], use_container_width=True)
                        st.caption(f"**Prompt:** {img_data['prompt']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download",
                                data=img_data['image_bytes'],
                                file_name=os.path.basename(img_data['filepath']),
                                mime="image/png",
                                use_container_width=True
                            )
                        with col2:
                            if st.button(f"🔄 Regenerate", key=f"regen_{idx}", use_container_width=True):
                                st.session_state.current_prompts = [img_data['prompt']]
                                st.session_state.generation_state = "regenerating"
                                st.rerun()
            else:
                st.info("Generated images will appear here...")
        
        # Sidebar
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Settings
            st.subheader("Generation Settings")
            num_var = st.number_input("Number of Variations", 1, 10, st.session_state.num_variations)
            if num_var != st.session_state.num_variations:
                st.session_state.num_variations = num_var
            
            russian = st.checkbox("Russian Text Requirement", st.session_state.russian_guardrail)
            if russian != st.session_state.russian_guardrail:
                st.session_state.russian_guardrail = russian
            
            st.divider()
            
            # Statistics
            st.subheader("📊 Session Stats")
            st.metric("Total Images", len(st.session_state.generated_images))
            st.metric("Generation Runs", st.session_state.generation_counter)
            
            if st.button("🗑️ Clear All", use_container_width=True):
                for key in ["messages", "generated_images", "current_prompts"]:
                    st.session_state[key] = []
                st.session_state.generation_counter = 0
                st.session_state.session_folder = None
                st.rerun()
            
            st.divider()
            
            # Instructions
            st.subheader("📖 Quick Guide")
            st.markdown("""
            1. Enter your API key
            2. Type an image prompt
            3. Choose to improve with GPT
            4. View & download images
            
            **Commands:**
            - `help` - Show commands
            - `improve` - Enhance prompts
            - `regenerate` - Same prompts
            - `adjust` - Modify prompts
            - `clear` - Clear chat
            """)


def main():
    """Main application entry point"""
    app = ImageGeneratorChat()
    app.render_ui()


if __name__ == "__main__":
    main()
