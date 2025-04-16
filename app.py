import os
import uuid
import logging
import requests
import sys
from flask import Flask, request, jsonify, render_template, send_file, url_for
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv
import time
from flask_cors import CORS  # Add CORS to handle cross-domain requests
import re  # Add module for regular expressions
import gc  # Add garbage collector for explicit resource release
import base64 # Add base64 import
import random # Add random import for random video selection

# Determine the base directory whether running as script or frozen executable
# This is primarily for bundled resources like static/templates
if getattr(sys, 'frozen', False):
    # Running as a PyInstaller bundle
    bundle_dir = sys._MEIPASS
else:
    # Running as a normal script
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths based on the bundle directory for bundled resources
STATIC_FOLDER_ABS = os.path.join(bundle_dir, 'static')
TEMPLATE_FOLDER_ABS = os.path.join(bundle_dir, 'templates')
AUDIO_FOLDER_ABS = os.path.join(STATIC_FOLDER_ABS, 'audio')
IMAGE_FOLDER_ABS = os.path.join(STATIC_FOLDER_ABS, 'images')

# Ensure necessary bundled resource folders exist (if needed, though usually handled by PyInstaller bundling)
# os.makedirs(AUDIO_FOLDER_ABS, exist_ok=True)
# os.makedirs(IMAGE_FOLDER_ABS, exist_ok=True)

# Determine the directory containing the executable or the script
if getattr(sys, 'frozen', False):
    # Path for files located next to the executable (like .env)
    app_dir = os.path.dirname(sys.executable)
else:
    # Path for files located next to the script
    app_dir = os.path.dirname(os.path.abspath(__file__))

# Logging configuration needs path relative to app_dir if writing logs there
log_file_path = os.path.join(app_dir, "app.log")

# Logging configuration with Unicode support for Windows
class UnicodeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # If encoding error occurs, use ASCII with replacement of non-printable characters
                stream.write(msg.encode('ascii', 'replace').decode('ascii') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Setup logging with a file logger
file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Console logger with unicode handling
console_handler = UnicodeStreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure the main logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False  # Prevent double logging

# Load environment variables from .env file located relative to the app path
dotenv_path = os.path.join(app_dir, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logger.warning(f".env file not found at: {dotenv_path}. Relying on system environment variables.")

# Initialize Flask application with absolute paths for bundled resources
app = Flask(__name__,
            static_folder=STATIC_FOLDER_ABS, # Bundled static
            template_folder=TEMPLATE_FOLDER_ABS) # Bundled templates
CORS(app)  # Activate CORS for all routes

# Global variables and constants
app.start_time = time.time()  # Set start time immediately
conversation_history = []
voiceChatActive = False  # Variable initialization
MAX_HISTORY_LENGTH = 10  # Reduce history size to prevent memory leaks
MAX_REQUEST_TIME = 30  # Maximum request execution time in seconds

# --- Use the new absolute paths for SAVED/SERVED dynamic content ---
AUDIO_FOLDER = os.path.join(app_dir, 'static', 'audio')
IMAGE_FOLDER = os.path.join(app_dir, 'static', 'images')
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
logger.info(f"Using persistent AUDIO_FOLDER: {AUDIO_FOLDER}")
logger.info(f"Using persistent IMAGE_FOLDER: {IMAGE_FOLDER}")

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STABILITYAI_API_KEY = os.getenv("STABILITYAI_API_KEY") # Add Stability AI key
# Get Google Search key and Search Engine ID
GOOGLE_CSE_ID = os.getenv("GOOGLE_URL").split("cx=")[-1] if os.getenv("GOOGLE_URL") else None
GOOGLECUSTOMSEARCH_API_KEY = os.getenv("GOOGLECUSTOMSEARCH_API_KEY")
YOUTUBEDATAV3_API_KEY = os.getenv("YOUTUBEDATAV3_API_KEY") # Not used yet
# DEEPAI_API_KEY is no longer needed
# DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY")

# Remove DEEPAI_API_KEY check
# if not OPENAI_API_KEY or not DEEPAI_API_KEY:
if not OPENAI_API_KEY:
    logger.error("OpenAI API key not set. Check environment variables.")
    raise EnvironmentError("OpenAI API key not set")
if not STABILITYAI_API_KEY:
    logger.warning("Stability AI API key not set. Image generation might fail.") # Warning, not an error
# Check Google Search keys
if not GOOGLE_CSE_ID or not GOOGLECUSTOMSEARCH_API_KEY:
    logger.warning("Google Custom Search API key or CSE ID not set. Image search will fail.")
# Check YouTube key
if not YOUTUBEDATAV3_API_KEY:
    logger.warning("YouTube Data API v3 key not set. Video search will fail.")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to detect text language
def detect_language(text):
    """
    Determines the language of the text based on character analysis.
    Returns the language code ('ru', 'en', 'fr') or 'other'.
    """
    try:
        if not text:
            logger.warning("Empty text for language detection, defaulting to 'other'")
            return "other"

        # Check for significant presence of Arabic script characters
        # Range U+0600 to U+06FF covers Arabic
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        arabic_count = len(arabic_pattern.findall(text))
        
        # Simple threshold: if more than 5 Arabic chars are found, classify as 'other'
        # This helps filter titles that might mix scripts but are primarily Arabic
        if arabic_count > 5: 
            logger.debug(f"Detected significant Arabic script in: '{text[:30]}...'")
            return "other" # Treat Arabic script presence as 'other'

        # Existing logic for Cyrillic, Latin, French
        cyrillic_pattern = re.compile(r'[а-яА-ЯёЁ]')
        latin_pattern = re.compile(r'[a-zA-Z]')
        french_chars = set('çéàèêëïîôùûüÿœæÇÉÀÈÊËÏÎÔÙÛÜŸŒÆ')

        cyrillic_count = len(cyrillic_pattern.findall(text))
        latin_count = len(latin_pattern.findall(text))
        french_char_count = sum(1 for char in text if char in french_chars)

        # Determine total relevant characters for proportion calculation
        total_known_chars = cyrillic_count + latin_count + arabic_count # Include arabic here for proportion check

        logger.debug(f"Text: '{text[:30]}...', cyrillic: {cyrillic_count}, latin: {latin_count}, french: {french_char_count}, arabic: {arabic_count}")

        # Check for Cyrillic dominance
        if cyrillic_count > 0 and cyrillic_count >= latin_count: 
             # Requires at least one cyrillic char and more cyrillic than latin
            return "ru"
        
        # Check for French (requires Latin presence)
        # Condition: French chars exist AND (more than 5 french chars OR latin count is dominant over cyrillic)
        elif french_char_count > 0 and latin_count > 0 and (french_char_count > 5 or latin_count >= cyrillic_count):
            return "fr"

        # Check for English (requires Latin presence, non-French dominance)
        # Condition: Latin chars exist AND latin count is dominant over cyrillic
        elif latin_count > 0 and latin_count >= cyrillic_count:
            return "en"
        
        else:
            # If none of the above dominate or only unknown scripts/symbols are present
            logger.debug(f"No dominant ru/en/fr script found in: '{text[:30]}...' Defaulting to 'other'.")
            return "other"

    except Exception as e:
        logger.error(f"Error detecting language: {e}", exc_info=True)
        return "other" # Default to 'other' on error

# Define command keywords in lowercase for different languages
PREFIX_KEYWORDS = {"чат", "chat", "tchat"}

IMAGE_KEYWORDS = {
    # Russian
    "нарисуй", "создай картинку", "изобрази", "покажи", "сделай картинку", "нарисуй мне", "сгенерируй изображение",
    # English
    "draw", "paint", "generate image", "create image", "show me image", "picture of",
    # French
    "dessine", "dessine-moi", "crée une image", "génère une image", "montre-moi une image", "image de"
}

VOICE_ON_KEYWORDS = {
    # Russian
    "приём", "активация",
    # English
    "activate", "voice on", "start voice",
    # French
    "activer", "activation", "voix on"
}

VOICE_OFF_KEYWORDS = {
    # Russian
    "отбой", "деактивация",
    # English
    "deactivate", "voice off", "stop voice",
    # French
    "désactiver", "désactivation", "voix off"
}

# New keywords for image search
FIND_IMAGE_KEYWORDS = {
    # Russian
    "картинка", "картинку", "найди картинку", "покажи картинку",
    # English
    "picture", "image", "find picture", "find image", "search picture", "search image",
    # French
    "image", "photo", "cherche image", "trouve image" # Use 'image' as international
}

# Keywords for video search
FIND_VIDEO_KEYWORDS = {
    # Russian
    "видео", "найди видео", "покажи видео", "клип", "ролик",
    # English
    "video", "find video", "search video", "show video", "clip",
    # French
    "vidéo", "cherche vidéo", "trouve vidéo", "montre vidéo", "clip"
}

# Blacklist of words/stems for video titles (lowercase)
VIDEO_TITLE_BLACKLIST = {
    # Religion/Spirituality (stems and words)
    "surah", "quran", "коран", "сура", "ayat", "аят", "tilawat", "аллах", "мусульм", "ислам",
    "allah", "muslim", "islam",
    "христос", "иисус", "библия", "церк", "евангелие", "святой", "молитв", "христиан", # 'church stem' instead of 'church'
    "christ", "jesus", "bible", "church", "gospel", "saint", "prayer", "christian",
    "тора", "талмуд", "синагог", "иудей", "иудаизм",
    "torah", "talmud", "synagogue", "jewish", "judaism",
    "бог", "религ", "духовн", "пророк", "вера", "дьявол", "сатан", # 'religion stem', 'spiritual stem'
    "god", "religion", "religious", "spiritual", "prophet", "faith", "devil", "satan",
    # Violence/Cruelty (stems and words)
    "убий", "убит", "кровь", "жесток", "насил", "драка", "тюрьм", "террор", # stems for 'kill', 'cruel', 'violent', 'prison', 'terror'
    "kill", "killing", "murder", "violent", "violence", "brutal", "jail", "prison", "fight", "gore", "blood", "terror", "terreur",
    # Horror/Fear (stems and words)
    "хоррор", "ужас", "страх", "страш", "призрак", "демон", "монстр", # stems for 'horror', 'scary'
    "horror", "scary", "fear", "ghost", "demon", "monster", "terrifying", "creepy",
    # Politics/News/War (stems and words)
    "полит", "новост", "выбор", "президент", "правительств", "войн", # stems for 'politic', 'news', 'election', 'government', 'war'
    "politic", "news", "election", "president", "government", "war", "conflict",
    # Adult Content/Themes
    "секс", "эротик", "для взрослых", "18+",
    "sex", "sexy", "erotic", "adult", "mature", "nsfw",
    # Dependencies/Drugs (stems and words)
    "казино", "ставк", "азарт", "рулетк", "покер", "нарко", # stems for 'bet', 'gamble', 'roulette', 'drug'
    "casino", "betting", "gambl", "poker", "drug", # 'gambl', 'drug'
    # Death/Injury (stems and words)
    "труп", "казн", "пытк", "смерт", # stems for 'execution', 'torture', 'death'
    "corpse", "execution", "torture", "death",
}

@app.route('/')
def home():
    """Отображение веб-интерфейса."""
    return render_template('chat.html')

def generate_image_logic(prompt: str) -> dict:
    """
    Логика генерации изображения с помощью Stability AI (SDXL).
    Возвращает словарь с ключом 'image_url' или 'error'.
    """
    if not STABILITYAI_API_KEY:
        return {'error': 'Stability AI API key is not configured.'}

    engine_id = "stable-diffusion-xl-1024-v1-0"
    api_host = os.getenv('API_HOST', 'https://api.stability.ai') # Can be overridden if necessary
    url = f"{api_host}/v1/generation/{engine_id}/text-to-image"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STABILITYAI_API_KEY}",
    }

    payload = {
        "text_prompts": [
            {
                "text": prompt
            }
        ],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30, # Default value for SDXL 1.0
    }

    try:
        logger.info(f"Stability AI generation request: '{prompt[:50]}...'")
        # Increase timeout, generation can be long
        response = requests.post(url, headers=headers, json=payload, timeout=90)

        if response.status_code != 200:
            # Try to get error message from API response
            error_message = f"Stability AI API error (Status {response.status_code})"
            try:
                error_details = response.json()
                # Look for errors in standard fields
                if 'message' in error_details:
                    error_message += f": {error_details['message']}"
                elif 'errors' in error_details:
                     error_message += f": {str(error_details['errors'])}"
                elif 'name' in error_details: # Some errors have a name field
                    error_message += f": {error_details['name']}"
                else:
                    error_message += f": {response.text[:100]}" # Return part of the response text
            except Exception:
                error_message += f": {response.text[:100]}"
            logger.error(error_message)
            return {'error': error_message}

        data = response.json()

        # Check for artifacts
        if "artifacts" not in data or not data["artifacts"]:
            logger.error("Stability AI API returned 200 but no artifacts found.")
            return {'error': 'No image artifacts received from Stability AI.'}

        # Process the first artifact
        for i, image in enumerate(data["artifacts"]):
            if image.get("base64") and image.get("finishReason") == 'SUCCESS':
                try:
                    img_data = base64.b64decode(image["base64"])
                    img_filename = f"{uuid.uuid4()}.png" # Generate a unique PNG file name
                    img_path = os.path.join(IMAGE_FOLDER, img_filename) # Uses absolute IMAGE_FOLDER

                    # Save the decoded image
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    logger.info(f"Stability AI image saved: {img_path}")

                    # Generate URL for file access via new endpoint
                    image_url = url_for('serve_image', filename=img_filename, _external=True)
                    return {'image_url': image_url}

                except base64.binascii.Error as decode_error:
                     logger.error(f"Error decoding base64 image from Stability AI: {decode_error}")
                     return {'error': 'Failed to decode image data.'}
                except IOError as save_error:
                    logger.error(f"Error saving image file from Stability AI: {save_error}")
                    return {'error': 'Failed to save image file.'}
                except Exception as process_error: # Catch other processing errors
                     logger.error(f"Error processing image artifact: {process_error}", exc_info=True)
                     return {'error': 'Error processing image data.'}
            else:
                 # Log the reason if not SUCCESS or no base64
                 reason = image.get("finishReason", "N/A")
                 has_base64 = "yes" if image.get("base64") else "no"
                 logger.warning(f"Skipping artifact {i}: finishReason={reason}, has_base64={has_base64}")

        # If loop finished without returning, no successful artifact was found
        logger.error("No successful image artifact with base64 data found in Stability AI response.")
        return {'error': 'No successful image generated by Stability AI.'}

    except requests.exceptions.Timeout:
        logger.error("Timeout when requesting Stability AI API")
        return {'error': "Stability AI API request timeout. Please try again."}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Error connecting to Stability AI API: {req_err}")
        return {'error': f'Could not connect to Stability AI: {req_err}'}
    except Exception as e:
        logger.error(f"Unexpected error during Stability AI image generation: {e}", exc_info=True)
        return {'error': f'Unexpected error generating image: {str(e)}'}

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервера."""
    # Add additional server status information
    memory_info = {}
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = {
            'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(interval=0.1)
        }
    except ImportError:
        memory_info = {"info": "psutil not installed, resource monitoring unavailable"}
    
    return jsonify({
        'status': 'ok', 
        'timestamp': time.time(),
        'uptime': time.time() - app.start_time,
        'system_info': memory_info,
        'history_length': len(conversation_history)
    }), 200

@app.route('/chat', methods=['POST'])
def chat():
    """Обработка текстовых сообщений от пользователя через веб-интерфейс."""
    global voiceChatActive, conversation_history

    request_start_time = time.time()
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] New request received from {request.remote_addr}")
    
    # History size limit
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        # Delete old messages, keeping only the latest ones
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]
        logger.info(f"[{request_id}] Conversation history truncated to {MAX_HISTORY_LENGTH} messages")

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning(f"[{request_id}] Invalid request: {data}")
            return jsonify({'error': 'Message field missing in request'}), 400

        user_input_original = data['message'].strip()
        if not user_input_original:
             logger.warning(f"[{request_id}] Empty user input after stripping")
             return jsonify({'error': 'Empty message.'}), 400

        user_input_lower = user_input_original.lower()
        logger.info(f"[{request_id}] User message: '{user_input_original[:80]}...' ({len(user_input_original)} chars)")

        command_processed = False
        potential_command_parts = user_input_lower.split(" ", 1)

        # Check if input starts with a known prefix
        if len(potential_command_parts) == 2 and potential_command_parts[0] in PREFIX_KEYWORDS:
            prefix_word = potential_command_parts[0]
            command_part = potential_command_parts[1]
            logger.debug(f"[{request_id}] Detected prefix '{prefix_word}', checking command part: '{command_part[:50]}...'")

            # Check for voice activation/deactivation (exact match of the part after prefix)
            if command_part in VOICE_ON_KEYWORDS:
                voiceChatActive = True
                logger.info(f"[{request_id}] Voice chat activated (via text command '{user_input_lower}')")
                command_processed = True
                return jsonify({'response': "Voice chat activated."}), 200
            elif command_part in VOICE_OFF_KEYWORDS:
                voiceChatActive = False
                logger.info(f"[{request_id}] Voice chat deactivated (via text command '{user_input_lower}')")
                command_processed = True
                return jsonify({'response': "Voice chat deactivated."}), 200

            # --- Check for FIND IMAGE command ---
            image_search_query = None
            for find_img_keyword in FIND_IMAGE_KEYWORDS:
                # Look for exact keyword match at the beginning of the command
                # Ensure there is a space after the word or it's the end of the string
                if command_part.startswith(find_img_keyword) and \
                   (len(command_part) == len(find_img_keyword) or command_part[len(find_img_keyword)] == ' '):
                    # Extract query text after the keyword
                    prompt_text = user_input_original[len(find_img_keyword):].strip()
                    if prompt_text:
                        image_search_query = prompt_text
                        logger.info(f"[{request_id}] Image SEARCH request detected with prefix '{prefix_word}' and keyword '{find_img_keyword}'")
                        break # Keyword found, exit the loop
                    else:
                        # Keyword exists, but query is missing
                        logger.warning(f"[{request_id}] Image search keyword '{find_img_keyword}' detected, but query is missing.")
                        # Can return an error or process as a normal message below
                        # For now, just log and continue, generation or normal chat might trigger
                        break 

            if image_search_query:
                result = find_google_image(image_search_query, detect_language(user_input_original)) # Call Google search
                logger.info(f"[{request_id}] Google Image Search result: {'success' if 'found_image_urls' in result else 'error'}")
                status = 200 if 'found_image_urls' in result else 500
                # Add text response if pictures are found
                if 'found_image_urls' in result:
                    # Simple message about successful search
                    response_text_map = {
                        'ru': f"Нашел {len(result['found_image_urls'])} картинок по запросу: {image_search_query[:30]}...",
                        'en': f"Found {len(result['found_image_urls'])} pictures for: {image_search_query[:30]}...",
                        'fr': f"J'ai trouvé {len(result['found_image_urls'])} images pour: {image_search_query[:30]}..."
                    }
                    result['response'] = response_text_map.get(detect_language(user_input_original), response_text_map['en'])
                # Otherwise, the error field should already be in the result from find_google_image
                
                command_processed = True # Command processed (successfully or with search error)
                 # Check if the total request timeout has been exceeded
                if time.time() - request_start_time > MAX_REQUEST_TIME:
                    logger.warning(f"[{request_id}] Request processing timeout exceeded after image search")
                    return jsonify({'error': 'Request processing timeout'}), 504
                return jsonify(result), status
            # --- End FIND IMAGE Check ---

            # --- Check for FIND VIDEO command ---
            video_search_query = None
            for find_video_keyword in FIND_VIDEO_KEYWORDS:
                # Look for exact video keyword match at the beginning of the command
                if command_part.startswith(find_video_keyword) and \
                   (len(command_part) == len(find_video_keyword) or command_part[len(find_video_keyword)] == ' '):
                    # Extract video query text after the keyword
                    # Use original text without prefix to preserve case
                    original_command_part_for_video = user_input_original.split(" ", 1)[1] if len(user_input_original.split(" ", 1)) > 1 else ""
                    prompt_text = original_command_part_for_video[len(find_video_keyword):].strip()
                    if prompt_text:
                        video_search_query = prompt_text
                        logger.info(f"[{request_id}] Video SEARCH request detected with prefix '{prefix_word}' and keyword '{find_video_keyword}'")
                        command_processed = True
                        break
                    else:
                        logger.warning(f"[{request_id}] Video search keyword '{find_video_keyword}' detected, but query is missing.")
                        command_processed = True # Processed, but with an error
                        return jsonify({'error': 'Please specify what video to search for.'}), 400
            
            if video_search_query:
                result = find_youtube_video(video_search_query) # Call YouTube search
                logger.info(f"[{request_id}] YouTube Video Search result: {'success' if 'found_video_details' in result else 'error'}")
                status = 200 if 'found_video_details' in result else 500
                # Add text response if video is found
                if 'found_video_details' in result:
                    response_text_map = {
                        'ru': f"Нашел видео по запросу '{result['found_video_details'][0]['video_title'][:30]}...'",
                        'en': f"Found a video for '{result['found_video_details'][0]['video_title'][:30]}...'",
                        'fr': f"J'ai trouvé une vidéo pour '{result['found_video_details'][0]['video_title'][:30]}...'"
                    }
                    result['response'] = response_text_map.get(detect_language(user_input_original), response_text_map['en'])
                # Otherwise, the error field should already be in the result from find_youtube_video
                
                 # Check if the total request timeout has been exceeded
                if time.time() - request_start_time > MAX_REQUEST_TIME:
                    logger.warning(f"[{request_id}] Request processing timeout exceeded after video search")
                    return jsonify({'error': 'Request processing timeout'}), 504
                return jsonify(result), status
            # --- End FIND VIDEO Check ---

            # Check for image generation command (starts with keyword after prefix)
            image_prompt = None
            for img_keyword in IMAGE_KEYWORDS:
                if command_part.startswith(img_keyword):
                    # Extract prompt text after the image keyword
                    # Need the original case input without the prefix
                    original_command_part = user_input_original.split(" ", 1)[1]
                    prompt_text = original_command_part[len(img_keyword):].strip()
                    if prompt_text:
                        image_prompt = prompt_text
                        logger.info(f"[{request_id}] Image generation request detected with prefix '{prefix_word}' and keyword '{img_keyword}'")
                        command_processed = True
                        break

            if image_prompt:
                result = generate_image_logic(image_prompt) # Pass extracted prompt
                logger.info(f"[{request_id}] Image request result: {'success' if 'image_url' in result else 'error'}")
                status = 200 if 'image_url' in result else 500
                
                # Check if timeout has been exceeded
                if time.time() - request_start_time > MAX_REQUEST_TIME:
                    logger.warning(f"[{request_id}] Request processing timeout exceeded")
                    return jsonify({'error': 'Request processing timeout'}), 504
                
                return jsonify(result), status

            # If prefix was detected but command part didn't match anything known
            if not command_processed:
                 logger.info(f"[{request_id}] Prefix '{prefix_word}' detected but command '{command_part}' is unknown. Treating as normal message.")
                 # Fall through to treat the whole original message as input

        # If no prefix detected or command wasn't processed, treat as standard message
        if not command_processed:
            conversation_history.append({"role": "user", "content": user_input_original})
            logger.info(f"[{request_id}] Processing as standard chat message.")

            # Request to OpenAI ChatCompletion considering history
            try:
                logger.info(f"[{request_id}] Sending request to OpenAI, history: {len(conversation_history)} messages")
                
                # Set timeout
                if time.time() - request_start_time > MAX_REQUEST_TIME * 0.6:
                    logger.warning(f"[{request_id}] Not enough time for API request")
                    return jsonify({'error': 'Not enough time to process request'}), 504
                
                # Format messages for API
                system_message = {"role": "system", "content": "Вы — помощник, учитель и воспитатель, который помогает детям от трёх лет учиться и развиваться, а также предостерегает их от ошибок и глупостей, объясняя причину. Отвечайте на русском, если вопрос на русском языке, отвечайте на английском, если вопрос на английском языке, отвечайте на французском, если вопрос на французском языке, используя простые и понятные формулировки. Задавайте вопросы в 5% случаев."}
                messages_to_send = [system_message] + conversation_history

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages_to_send,
                    max_tokens=500,
                    temperature=0.7,
                    # Add timeout for the API request itself
                    timeout=MAX_REQUEST_TIME * 0.5 # Approximately half of the total time
                )
                api_request_time = time.time() - (request_start_time + (MAX_REQUEST_TIME * 0.6)) # Rough estimate
                logger.info(f"[{request_id}] OpenAI API request took ~{api_request_time:.2f} seconds")

            except openai.APIConnectionError as e:
                logger.error(f"[{request_id}] OpenAI API connection error: {e}")
                return jsonify({'error': f'Failed to connect to OpenAI API: {e}'}), 503 # Service Unavailable
            except openai.RateLimitError as e:
                logger.error(f"[{request_id}] OpenAI API rate limit exceeded: {e}")
                return jsonify({'error': f'OpenAI API rate limit reached. Please try again later.'}), 429 # Too Many Requests
            except openai.APITimeoutError as e:
                logger.error(f"[{request_id}] OpenAI API request timed out: {e}")
                return jsonify({'error': f'OpenAI API request timed out.'}), 504 # Gateway Timeout
            except openai.APIError as e: # General API error
                logger.error(f"[{request_id}] OpenAI API returned an error: {e}")
                return jsonify({'error': f'OpenAI API error: {e}'}), 502 # Bad Gateway
            except Exception as api_error: # Other errors during request
                logger.error(f"[{request_id}] Error during OpenAI API request: {api_error}", exc_info=True)
                return jsonify({'error': f'Error requesting OpenAI API: {str(api_error)}'}), 500

            if response and response.choices:
                # Check for presence and type of content
                if not response.choices[0].message or not response.choices[0].message.content:
                    logger.error(f"[{request_id}] OpenAI response message content is empty or invalid.")
                    return jsonify({'error': 'Received empty response from AI model.'}), 500

                gpt_response = response.choices[0].message.content.strip()
                logger.info(f"[{request_id}] Response from OpenAI: '{gpt_response[:80]}...' ({len(gpt_response)} chars)")

                # Add model response to history
                conversation_history.append({"role": "assistant", "content": gpt_response})

                # Determine response language for correct pronunciation
                detected_lang = detect_language(gpt_response)
                logger.info(f"[{request_id}] Detected response language: {detected_lang}")

                # Check if timeout has been exceeded
                if time.time() - request_start_time > MAX_REQUEST_TIME * 0.8:
                    logger.warning(f"[{request_id}] Not enough time for audio generation")
                    return jsonify({'response': gpt_response, 'language': detected_lang, 'error_audio': 'Timeout before audio generation'}), 200

                # Generate audio file using gTTS with the determined language
                try:
                    audio_filename = f"{str(uuid.uuid4())}.mp3"
                    audio_path = os.path.join(AUDIO_FOLDER, audio_filename) # Uses absolute AUDIO_FOLDER

                    tts = gTTS(text=gpt_response, lang=detected_lang)
                    tts.save(audio_path)
                    logger.info(f"[{request_id}] MP3 saved: {audio_path} in language {detected_lang}")

                    # Forming the correct URL to access the audio file
                    audio_url = url_for('serve_audio', filename=audio_filename, _external=True)
                    
                    request_time = time.time() - request_start_time
                    logger.info(f"[{request_id}] Request processed in {request_time:.2f} seconds")
                    
                    return jsonify({
                        'response': gpt_response,
                        'audio_url': audio_url,
                        'language': detected_lang,
                        'request_time': round(request_time, 2)
                    })
                except Exception as audio_error:
                    logger.error(f"[{request_id}] Error creating audio: {audio_error}")
                    # Return response without audio if audio file creation failed
                    return jsonify({'response': gpt_response, 'language': detected_lang, 'error_audio': str(audio_error)}), 200
            else:
                logger.error(f"[{request_id}] GPT gave no response or invalid response")
                return jsonify({'error': 'No response from GPT'}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error processing request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Returns the generated audio file."""
    # Use the absolute AUDIO_FOLDER path
    audio_file = os.path.join(AUDIO_FOLDER, filename)
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return jsonify({'error': 'File not found'}), 404
    # Specify mimetype for better compatibility
    return send_file(audio_file, mimetype="audio/mpeg")

# Add route to serve saved images
@app.route('/static/images/<filename>')
def serve_image(filename):
    """Returns the saved image."""
    # Use the absolute IMAGE_FOLDER path
    image_file = os.path.join(IMAGE_FOLDER, filename)
    if not os.path.exists(image_file):
        logger.error(f"Image file not found: {image_file}")
        return jsonify({'error': 'File not found'}), 404
    # Try to determine mimetype or use standard for png
    mimetype = 'image/png' # Default to PNG as we save in PNG
    # try:
    #     import mimetypes
    #     mimetype = mimetypes.guess_type(image_file)[0] or 'image/png'
    # except ImportError:
    #     pass
    return send_file(image_file, mimetype=mimetype)

# Add route to clean up old audio files
@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Cleanup of old temporary files."""
    try:
        # Use the absolute AUDIO_FOLDER path
        audio_files = os.listdir(AUDIO_FOLDER)
        count = 0
        total_size = 0
        
        # Delete files older than a certain time
        for file in audio_files:
            # Use the absolute AUDIO_FOLDER path
            file_path = os.path.join(AUDIO_FOLDER, file)
            # Check file creation time
            file_age = time.time() - os.path.getctime(file_path)
            # If the file is older than 30 minutes (1800 seconds), delete it
            if file_age > 1800:
                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    os.remove(file_path)
                    count += 1
                except OSError as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
        
        logger.info(f"Files cleaned: removed {count}, freed {total_size/1024:.2f} KB")
        return jsonify({'success': True, 'deleted_files': count, 'freed_space_kb': total_size/1024}), 200
    except Exception as e:
        logger.error(f"Error cleaning files: {e}")
        return jsonify({'error': str(e)}), 500

# Add route to clear conversation history
@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear message history."""
    global conversation_history
    try:
        conversation_history = []
        logger.info("Conversation history cleared")
        return jsonify({'success': True, 'message': 'Conversation history cleared'}), 200
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({'error': str(e)}), 500

# Run file cleanup on initialization without application context
def cleanup_on_startup():
    """Cleanup files on startup."""
    try:
        # Use the absolute AUDIO_FOLDER path
        audio_files = os.listdir(AUDIO_FOLDER)
        count = 0
        total_size = 0
        
        for file in audio_files:
            # Use the absolute AUDIO_FOLDER path
            file_path = os.path.join(AUDIO_FOLDER, file)
            file_age = time.time() - os.path.getctime(file_path)
            if file_age > 1800:
                try:
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    os.remove(file_path)
                    count += 1
                except OSError as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
        
        logger.info(f"Startup cleanup: removed {count}, freed {total_size/1024:.2f} KB")
    except Exception as e:
        logger.error(f"Error during startup cleanup: {e}")

# New function for searching images via Google Custom Search API
def find_google_image(query: str, lang: str) -> dict:
    """
    Ищет изображение через Google Custom Search API.
    Возвращает словарь с ключом 'found_image_urls' или 'error'.
    """
    if not GOOGLECUSTOMSEARCH_API_KEY or not GOOGLE_CSE_ID:
        return {'error': 'Google Custom Search is not configured on the server.'}

    # Remove or comment out adding clarifying words
    # if lang == 'ru':
    #     search_query = f"{query} для детей рисунок мультяшный"
    # elif lang == 'fr':
    #     search_query = f"{query} pour enfants dessin animé"
    # else: # Default to English
    #     search_query = f"{query} for kids cartoon drawing"
    # logger.info(f"Google Image Search request: '{search_query[:60]}...' (Original: '{query[:60]}...')")
    # Use the original query directly
    search_query = query
    logger.info(f"Google Image Search request: '{search_query[:60]}...'")

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLECUSTOMSEARCH_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'q': search_query,
        'searchType': 'image',
        'safe': 'active', # Enable strict safe search
        'num': 5, # Request 5 results to choose up to 3
        'lr': 'lang_ru|lang_en|lang_fr' # Limit result languages
    }

    try:
        response = requests.get(search_url, params=params, timeout=15)
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        results = response.json()
        items = results.get('items', [])
        
        image_urls = []
        if items:
            for item in items:
                link = item.get('link')
                if link:
                    image_urls.append(link)
                    if len(image_urls) == 3: # Collect no more than 3 links
                        break 
            
            if image_urls:
                logger.info(f"Google Image Search found {len(image_urls)} images, first: {image_urls[0][:60]}...")
                # Return the list of URLs
                return {'found_image_urls': image_urls} 
            else:
                logger.warning("Google Image Search items found, but no valid links.")
                return {'error': 'Found image items had no links.'}
        else:
            logger.info(f"Google Image Search returned no results for: '{search_query[:60]}...'")
            return {'error': 'No suitable images found for your request.'}

    except requests.exceptions.Timeout:
        logger.error("Timeout when requesting Google Custom Search API")
        return {'error': "Google Image Search request timeout."}
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error requesting Google Custom Search API: {http_err} - Response: {response.text[:200]}")
        # Try to extract the Google error message
        error_message = f"Google Search API HTTP error: {http_err}"
        try:
            error_detail = response.json().get('error', {}).get('message', '')
            if error_detail: error_message += f" - {error_detail}"
        except Exception: pass
        return {'error': error_message}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Error connecting to Google Custom Search API: {req_err}")
        return {'error': f'Could not connect to Google Search: {req_err}'}
    except Exception as e:
        logger.error(f"Unexpected error during Google Image Search: {e}", exc_info=True)
        return {'error': f'Unexpected error searching image: {str(e)}'}

# New function for searching videos on YouTube
def find_youtube_video(query: str) -> dict:
    """
    Ищет видео на YouTube через YouTube Data API v3.
    Возвращает словарь с ключом 'found_video_details' (список словарей) или 'error'.
    Каждый словарь в списке содержит 'youtube_embed_url' и 'video_title'.
    """
    if not YOUTUBEDATAV3_API_KEY:
        return {'error': 'YouTube Data API v3 key is not configured.'}

    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'key': YOUTUBEDATAV3_API_KEY,
        'maxResults': 25, # Request more for randomization
        'type': 'video',
        'safeSearch': 'strict' 
    }

    logger.info(f"YouTube Video Search request: '{query[:60]}...'")

    try:
        response = requests.get(search_url, params=params, timeout=15)
        response.raise_for_status() # Check for HTTP errors

        results = response.json()
        items = results.get('items', [])
        
        valid_videos = [] # Collect ALL valid videos
        if items:
            for item in items:
                video_id = item.get('id', {}).get('videoId')
                video_title = item.get('snippet', {}).get('title')
                # Check that ID and title exist
                if video_id and video_title:
                    video_title_lower = video_title.lower()
                    # 1. Check against blacklist words
                    if any(term in video_title_lower for term in VIDEO_TITLE_BLACKLIST):
                        logger.debug(f"Skipped video due to blacklist term: '{video_title[:40]}...'")
                        continue # Skip this video

                    # 2. Check title language (only if not blacklisted)
                    detected_lang = detect_language(video_title)
                    if detected_lang in ['ru', 'en', 'fr']:
                        embed_url = f"https://www.youtube.com/embed/{video_id}"
                        valid_videos.append({
                            'youtube_embed_url': embed_url, 
                            'video_title': video_title
                        })
                        logger.debug(f"Accepted video: '{video_title[:40]}...' (lang: {detected_lang})")
                    else:
                        logger.debug(f"Skipped video due to language: '{video_title[:40]}...' (lang: {detected_lang})")
            
            if valid_videos:
                # Select 3 random videos (or fewer if less found)
                num_to_select = min(len(valid_videos), 3)
                selected_videos = random.sample(valid_videos, num_to_select)
                
                logger.info(f"YouTube Video Search found {len(valid_videos)} valid videos, selected {len(selected_videos)}. First title: '{selected_videos[0]['video_title'][:40]}...'")
                # Return the list of randomly selected dictionaries
                return {'found_video_details': selected_videos} 
            else:
                logger.warning("YouTube Search returned items, but no valid videoId found.")
                return {'error': 'Could not find valid video details in search results.'}
        else:
            logger.info(f"YouTube Search returned no results for: '{query[:60]}...'")
            return {'error': 'No suitable video found for your request.'}

    except requests.exceptions.Timeout:
        logger.error("Timeout when requesting YouTube Data API")
        return {'error': "YouTube API request timeout."}
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error requesting YouTube Data API: {http_err} - Response: {response.text[:200]}")
        error_message = f"YouTube API HTTP error: {http_err}"
        try:
            error_detail = response.json().get('error', {}).get('message', '')
            if error_detail: error_message += f" - {error_detail}"
        except Exception: pass
        return {'error': error_message}
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Error connecting to YouTube Data API: {req_err}")
        return {'error': f'Could not connect to YouTube API: {req_err}'}
    except Exception as e:
        logger.error(f"Unexpected error during YouTube Video Search: {e}", exc_info=True)
        return {'error': f'Unexpected error searching video: {str(e)}'}

if __name__ == '__main__':
    try:
        # Perform cleanup without application context
        cleanup_on_startup()
        logger.info("Flask server starting...")
        
        # Disable reloader when in debug mode
        app.run(debug=False, host='0.0.0.0', use_reloader=False) # Ensure debug=False for distribution
    except Exception as e:
        logger.critical(f"Server startup error: {e}", exc_info=True)
