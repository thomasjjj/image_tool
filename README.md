# AI Image Generator - Streamlit Chat Interface

A Streamlit-based chat interface for AI image generation using OpenAI's DALL-E API. This app provides a terminal-style chat interface that mimics command-line interaction while offering a user-friendly web interface.

## Features

- 🎨 **AI Image Generation**: Generate images using OpenAI's DALL-E API
- 💬 **Chat Interface**: Terminal-style chat interface for natural interaction
- 🤖 **GPT-Powered Prompt Enhancement**: Automatically improve prompts using GPT-4
- 🔄 **Concurrent Generation**: Generate multiple image variations simultaneously
- 📁 **Automatic File Management**: Saves images and metadata automatically
- 🔑 **Secure API Key Input**: API key input on each session (not stored)
- 📥 **Download Support**: Download generated images directly from the interface

## Installation

1. **Clone or download the files**:
   - `streamlit_image_generator.py` - Main application file
   - `requirements.txt` - Python dependencies

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run streamlit_image_generator.py
   ```

4. **Open in browser**:
   The app will automatically open in your default browser, or navigate to:
   ```
   http://localhost:8501
   ```

## Usage

### Getting Started

1. **Enter API Key**: 
   - Enter your OpenAI API key in the password field at the top
   - Click "Set API Key" to activate the interface
   - The key is required for each session and is not stored

2. **Generate Images**:
   - Type an image prompt in the chat input
   - Choose whether to improve the prompt with GPT
   - View generated images in the right panel

### Available Commands

- **Image Prompt**: Any descriptive text for image generation
- `help` - Display available commands
- `improve` - Use GPT to improve the current prompts
- `regenerate` - Generate more images with the same prompts
- `adjust` - Modify the current prompts
- `clear` - Clear chat history and images
- `quit` / `exit` - End the current session

### Chat Flow Example

1. **User**: "A futuristic city at sunset"
2. **Assistant**: "Would you like me to improve this prompt with GPT?"
3. **User**: "yes"
4. **Assistant**: Generates 3 improved prompt variations
5. **System**: Displays generation progress and results
6. **Images**: Appear in the right panel with download buttons

## File Structure

```
generated_images/
├── YYYYMMDD_HHMMSS/
│   ├── 01_prompt_variation.png
│   ├── 02_prompt_variation.png
│   ├── 03_prompt_variation.png
│   └── generation_info.txt
```

## Features in Detail

### Terminal-Style Interface
- Displays commands and system messages in a terminal-like format
- Color-coded messages for different types of output
- Real-time progress updates during image generation

### Prompt Improvement
- GPT-4 analyzes your prompts and suggests improvements
- Generates multiple variations for diversity
- Maintains context throughout the conversation

### Image Management
- Automatically saves images to organized folders
- Creates metadata files for each generation session
- Provides download buttons for each generated image

### Concurrent Generation
- Uses async operations for faster multi-image generation
- Progress tracking for each image
- Error handling for failed generations

## Configuration

### API Settings
- Model: `dall-e-3` (can be modified in code)
- Size: `1024x1024` (default)
- Quality: `standard` (can be changed to `hd`)

### GPT Settings
- Model: `gpt-4` (for prompt improvements)
- Temperature: `0.8` (for creative variations)

## Troubleshooting

### Common Issues

1. **API Key Error**:
   - Ensure your OpenAI API key is valid
   - Check that you have DALL-E API access

2. **Generation Failures**:
   - Verify prompt doesn't violate content policies
   - Check API rate limits and quotas

3. **File Save Issues**:
   - Ensure write permissions in the application directory
   - Check available disk space

### Error Messages

- The app displays detailed error messages in the chat
- System messages appear with color coding:
  - 🟡 Yellow: Commands
  - 🔵 Cyan: System info
  - 🔴 Red: Errors
  - 🟢 Green: Success

## Notes

- API keys are never stored and must be entered for each session
- Generated images are saved locally in the `generated_images/` folder
- Metadata includes prompts, timestamps, and generation parameters
- The interface maintains chat history during the session

## Requirements

- Python 3.8+
- OpenAI API key with DALL-E access
- Internet connection for API calls
- Modern web browser

## Security

- API keys are handled securely in memory only
- No data is stored between sessions
- All files are saved locally on your machine

## Support

For issues or questions:
1. Check the error messages in the chat interface
2. Verify your API key and permissions
3. Ensure all dependencies are installed correctly
4. Check the `generated_images/` folder for saved outputs

---

Enjoy creating AI-generated images with this interactive chat interface! 🎨✨
