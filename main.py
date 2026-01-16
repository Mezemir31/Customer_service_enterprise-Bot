import sys
import os
import logging
import traceback
from google import genai
from config import Config
from database import init_database, update_database_schema
from bot import (
    GrievanceAnalyzer, 
    ResponseGenerator, 
    KeywordSuggestions, 
    ConversationManager,
    generate_session_id,
    display_analysis_results,
    log_conversation,
    show_quick_help,
    validate_user_input,
    analyze_with_retry,
    text_extractor,
    AICorrector
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BotApplication:
    
    def __init__(self):
        self.client = None
        self.spelling_corrector = None
        self.analyzer = None
        self.response_generator = None
        self.keyword_suggestions = None
        self.conversation_manager = None
        self.session_id = None
        self.session_context = None

    def initialize(self):
        """Initializeconnections."""
        print("Bot Initialization")
        print("-" * 30)
        
        # Initialize database
        init_database()
        update_database_schema()
        
        # Initialze for Gemini 
        try:
            if Config.GEMINI_API_KEY:
                self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
                logger.info("Gemini API client initialized")
                self.spelling_corrector = AICorrector(self.client)
            else:
                logger.warning(" No API Key found. Running in offline mode.")
        except Exception as e:
            logger.error(f"Gemini API client initialization failed: {e}")
        
        # Initialize bot components
        self.analyzer = GrievanceAnalyzer(self.client)
        self.response_generator = ResponseGenerator(self.client)
        self.keyword_suggestions = KeywordSuggestions()
        self.conversation_manager = ConversationManager()
        
        self._show_intro()
        
        self.session_id = generate_session_id()
        self.session_context = self.conversation_manager.get_session_context(self.session_id)
        
        if self.spelling_corrector:
            print("AI Spelling: ACTIVE")

    def _show_intro(self):
        if self.client:
            print("Customer Service Bot - Active")
            print("Integrated with Gemini API for advanced query processing.")
            print("\nNote: This is a simulation. No real NID data is accessed.")
        else:
            print("Running in BASIC MODE - AI features disabled")
        
        show_quick_help()

    def run(self):
        self.initialize()
        conversation_count = 0
        
        while True:
            try:
                user_input = input("\nDescribe your issue: ").strip()
                conversation_count += 1
                
                if self._handle_commands(user_input):
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                    continue
                
                self._process_input(user_input)
                
            except KeyboardInterrupt:
                print("\nSession interrupted.")
                break
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                print("Error occurred. Type 'suggest' for keywords.")
                logger.debug(f"🔧 Debug: {traceback.format_exc()}")
                continue

    def _handle_commands(self, user_input):
        cmd = user_input.lower()
        if cmd in ['quit', 'exit', 'bye']:
            print("Thank you for using Customer Support. Goodbye!")
            return True
        
        if cmd in ['suggest', 'keywords']:
            self.keyword_suggestions.display_suggestions()
            return True
            
        if cmd == 'help':
            show_quick_help()
            return True
            
        if cmd == 'history':
            self._show_history()
            return True
            
        if cmd == 'clear':
            self.session_id = generate_session_id()
            self.session_context = self.conversation_manager.get_session_context(self.session_id)
            print("New conversation started!")
            return True
            
        if cmd == 'test keywords':
            self._test_keywords()
            return True
            
        if cmd == 'spelltest' and self.spelling_corrector:
            self._test_spelling()
            return True
            
        if cmd == 'debug':
            self._debug_info(user_input)
            return True
            
        return False

    def _show_history(self):
        print("\nConversation History:")
        history = self.session_context['conversation_history'][-5:]
        if history:
            for i, msg in enumerate(history, 1):
                print(f"{i}. User: {msg['user_input'][:50]}...")
                print(f"   Bot: {msg['response'][:50]}...")
        else:
            print("No conversation history yet.")

    def _test_keywords(self):
        print("\n🧪 Testing keyword suggestions...")
        test_inputs = ["sms", "card", "change", "website", "how", "complaint", "foreign", "fan"]
        for test_input in test_inputs:
            print(f"\nTesting: '{test_input}'")
            self.keyword_suggestions.display_suggestions(test_input)

    def _test_spelling(self):
        print("\n Spelling Correction Test Mode")
        test_phrases = [
            "i want to chanj my name",
            "sms not comng",
            "kard printing problm",
            "የካርድ ማተም ችግር አለ",
            "maqaa koo jijjiiruu fedha"
        ]
        for phrase in test_phrases:
            lang = text_extractor._detect_language_fallback(phrase)
            corrected = self.spelling_corrector.correct_spelling_ai(phrase, lang)
            print(f"   '{phrase}' → '{corrected}'")

    def _debug_info(self, user_input):
        print("\n DEBUG INFORMATION:")
        print(f"Input: '{user_input}'")
        print(f"Lowercase: '{user_input.lower()}'")
        print("Keyword detection test:")
        test_keywords = ['fan', 'fin', 'uin', 'fcn', 'sms', 'resend']
        for kw in test_keywords:
            if kw in user_input.lower():
                print(f"Found keyword: {kw}")
        print("Current suggestions:")
        self.keyword_suggestions.display_suggestions(user_input)

    def _process_input(self, user_input):
        # INTELLIGENT DETECTION: If user sends only phone number or 29-digit number
        cleaned_input = user_input.strip()
        
        # Extract entities first to check what the user provided
        extracted_data = text_extractor.extract_with_tolerance(cleaned_input)
        phones = extracted_data.get('phones', [])
        digits = extracted_data.get('digits', [])
        fin_uin = extracted_data.get('fin_uin', [])
        fan_fcn = extracted_data.get('fan_fcn', [])
        
        # Check if user sent ONLY numbers (phone or 29-digit) without other text
        has_only_numbers = (
            len(cleaned_input) <= 30 and  # Reasonable length for numbers only
            any([phones, digits, fin_uin, fan_fcn]) and  # Has at least one number type
            not any(char.isalpha() for char in cleaned_input) and  # No letters
            len([item for sublist in [phones, digits, fin_uin, fan_fcn] for item in sublist]) == 1  # Only one number
        )
        
        # Check if it's a simple phone number or registration number
        is_simple_phone_or_digit = (
            (len(phones) == 1 and len(cleaned_input) <= 15) or  # Just a phone number
            (len(digits) == 1 and len(cleaned_input) == 29) or  # Just a 29-digit number
            (len(fin_uin) == 1 and len(cleaned_input) == 12) or  # Just a FIN/UIN
            (len(fan_fcn) == 1 and len(cleaned_input) == 16)     # Just a FAN/FCN
        )
        
        # If user sent only numbers, assume they need SMS resend
        if has_only_numbers or is_simple_phone_or_digit:
            print("🔍 Detected phone number or registration ID - assuming SMS resend request...")
            
            # Create artificial analysis for SMS resend
            analysis = {
                'language': 'english',
                'category': 'SMS_Resend_Confirmation',
                'subcategory': 'automatic_detection',
                'confidence': 'high',
                'sentiment': 3,
                'urgency': 'medium',
                'summary': 'User provided phone number or registration ID for SMS resend',
                'original_text': user_input,
                'corrected_text': user_input,
                'extracted_phones': phones,
                'extracted_digits_list': digits,
                'extracted_digits': digits[0] if digits else None,
                'extracted_fin_uin_list': fin_uin,
                'extracted_fin_uin': fin_uin[0] if fin_uin else None,
                'extracted_fan_fcn_list': fan_fcn,
                'extracted_fan_fcn': fan_fcn[0] if fan_fcn else None,
                'extracted_names': [],
                'extracted_keywords': ['sms', 'resend'],
                'ai_understanding': {
                    'intent': 'sms_issue',
                    'confidence': 'high',
                    'key_entities': ['phone', 'registration'],
                    'corrected_intent': 'User needs SMS confirmation resend'
                }
            }
            
            # Generate SMS resend response
            response = self.response_generator._handle_sms_resend(analysis, 'english')
            
        else:
            # Normal analysis flow for regular text input
            # Input validation
            is_valid, validation_msg = validate_user_input(user_input)
            if not is_valid:
                print(f"{validation_msg}")
                self.keyword_suggestions.display_suggestions(user_input)
                return
            
            # Enhanced analysis with AI-powered spelling correction
            print("Analyzing your issue with AI-powered...")
            
            # Pre-process input to detect common keywords
            user_input_lower = user_input.lower()
            if any(keyword in user_input_lower for keyword in ['fan', 'fin', 'uin', 'fcn', 'sms']):
                print("Detected SMS/FIN/FAN related query...")

            analysis = analyze_with_retry(
                self.analyzer,
                user_input, 
                self.session_id, 
                self.session_context['conversation_history']
            )

            # Fix: Ensure language is properly handled
            if isinstance(analysis.get('language'), dict):
                analysis['language'] = analysis['language'].get('language', 'english')
            
            # If analysis went to Others but we detected relevant keywords, force a better category
            if (analysis['category'] == 'Others' and 
                any(keyword in user_input_lower for keyword in ['fan', 'fin', 'uin', 'fcn', 'sms', 'resend'])):
                print("Re-categorizing as SMS resend request...")
                analysis['category'] = 'SMS_Resend_Confirmation'
                analysis['subcategory'] = 'resend_request'
                analysis['confidence'] = 'medium'
                analysis['summary'] = 'User needs SMS/FAN/FIN resend or assistance'
            
            # Intelligent response generation based on category
            response = self.response_generator.generate_response(analysis, self.session_context)
        
        # Display results
        display_analysis_results(analysis)
        print(f"\n[#]The RESPONSE, can be customised as needed ({analysis['language']}):\n{response}")
        
        # Update conversation
        primary_phone = analysis['extracted_phones'][0] if analysis['extracted_phones'] else None
        log_conversation(self.session_id, primary_phone, user_input, analysis, response)
        self.conversation_manager.add_interaction(self.session_context, user_input, analysis, response)

if __name__ == "__main__":
    app = BotApplication()
    app.run()
