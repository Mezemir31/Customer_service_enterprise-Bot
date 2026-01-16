import datetime
import time
import json
import os
import hashlib
import sqlite3
from typing import Dict, List, Tuple, Optional
from google import genai
from config import Config
from database import get_db_connection
from text_processing import TextExtractor, AICorrector, SpellingTolerance

# Initialize text extractor
text_extractor = TextExtractor()

class GrievanceAnalyzer:
    """
    Analyzes user grievances to determine category, sentiment, urgency, and extract entities.
    Uses a hybrid approach of regex matching and Gemini API analysis.
    """
    def __init__(self, client):
        """
        Initialize the analyzer.
        
        Args:
            client: The Gemini API client instance.
        """
        self.client = client
        self.fallback_mode = False
        self.spelling_tolerance = SpellingTolerance()
        self.ai_corrector = AICorrector(client) if client else None
    
    def analyze_grievance(self, message: str, session_id: str, conversation_history: List = None) -> Dict:
        """
        Analyze a grievance message.
        
        Args:
            message: The user's input message.
            session_id: The current session ID.
            conversation_history: List of previous interactions.
            
        Returns:
            Dict containing analysis results (category, sentiment, urgency, extracted entities, etc.).
        """
        # Step 1: Basic language detection
        language = self._detect_language_fallback(message)
        
        # Step 2: Multi-layer spelling correction
        corrected_message = self._multi_layer_correction(message, language)
        
        # Step 3: Try AI analysis with corrected text
        if self.client and not self.fallback_mode:
            try:
                analysis = self._try_ai_analysis(corrected_message, conversation_history)
                if analysis and analysis.get('category') != 'AI_API_Error':
                    # Enhance with AI understanding
                    analysis = self._add_ai_context(analysis, message, corrected_message, language)
                    return analysis
            except Exception as e:
                print(f"Analysis failed, using fallback: {e}")
                self.fallback_mode = True
        
        # Step 4: Enhanced fallback with AI correction
        return self._fallback_analysis(message, corrected_message, language)
    
    def _multi_layer_correction(self, message: str, language: str) -> str:
        """Apply multiple layers of spelling correction"""
        # Layer 1: Basic normalization
        normalized = self.spelling_tolerance.normalize_text(message, language)
        
        # Layer 2: AI-powered correction if available
        if self.ai_corrector and len(message) > 3:  # Only use AI for substantial text
            try:
                ai_corrected = self.ai_corrector.correct_spelling_ai(normalized, language)
                return ai_corrected
            except Exception as e:
                print(f"Correction failed, using normalized: {e}")
        
        return normalized
    
    def _add_ai_context(self, analysis: Dict, original: str, corrected: str, language: str) -> Dict:
        """Enhance analysis with AI understanding of intent"""
        if not self.ai_corrector:
            return analysis
        
        try:
            ai_understanding = self.ai_corrector.enhance_understanding_ai(original, corrected, language)
            
            # Update analysis with AI insights
            analysis['ai_understanding'] = ai_understanding
            analysis['confidence'] = self._combine_confidence(analysis.get('confidence', 'medium'), 
                                                            ai_understanding['confidence'])
            
            # If AI has high confidence in a different intent, consider updating category
            if (ai_understanding['confidence'] == 'high' and 
                ai_understanding['intent'] != 'other' and
                analysis.get('confidence') == 'low'):
                
                new_category = self._map_intent_to_category(ai_understanding['intent'])
                if new_category != analysis['category']:
                    print(f" AI overriding category: {analysis['category']} → {new_category}")
                    analysis['category'] = new_category
                    analysis['summary'] = f"AI-corrected: {ai_understanding['corrected_intent']}"
            
            return analysis
            
        except Exception as e:
            print(f"AI understanding enhancement failed: {e}")
            return analysis
    
    def _combine_confidence(self, conf1: str, conf2: str) -> str:
        """Combine confidence levels from different systems"""
        confidence_levels = {'low': 0, 'medium': 1, 'high': 2}
        combined = (confidence_levels.get(conf1, 1) + confidence_levels.get(conf2, 1)) / 2
        
        if combined >= 1.5:
            return 'high'
        elif combined >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _map_intent_to_category(self, intent: str) -> str:
        """Map AI intent to our category system"""
        intent_map = {
            'sms_issue': 'SMS_Resend_Confirmation',
            'card_status': 'Card_Printing',
            'data_update': 'Data_Update_Request',
            'account_help': 'Account_Management',
            'payment': 'Payment_Issues',
            'technical': 'Technical_Issues',
            'information': 'Information_Request',
            'complaint': 'Service_Complaint',
            'registration': 'Registration_Process'
        }
        return intent_map.get(intent, 'Others')
    
    def _fallback_analysis(self, original_message: str, corrected_message: str, language: str) -> Dict:
        """Enhanced fallback with AI-powered understanding"""
        print("Using fallback analysis...")
        
        # Ensure language is a string
        if isinstance(language, dict):
            language = language.get('language', 'english')
        
        corrected_lower = corrected_message.lower()
        
        # Get AI understanding even in fallback mode
        ai_understanding = None
        if self.ai_corrector:
            try:
                ai_understanding = self.ai_corrector.enhance_understanding_ai(
                    original_message, corrected_message, language
                )
                print(f"AI Intent: {ai_understanding}")
            except Exception as e:
                print(f"Fallback AI understanding failed: {e}")
        
        # Use AI intent if available and confident
        if ai_understanding and ai_understanding['confidence'] == 'high':
            category = self._map_intent_to_category(ai_understanding['intent'])
        else:
            category = self._detect_category(corrected_lower, language)
        
        extraction = text_extractor.extract_with_tolerance(original_message)
        
        analysis = {
            'language': language,
            'category': category,
            'subcategory': self._detect_subcategory(category, corrected_lower),
            'confidence': ai_understanding['confidence'] if ai_understanding else 'medium',
            'sentiment': self._detect_sentiment(corrected_message),
            'urgency': self._detect_urgency(corrected_lower),
            'summary': f"AI-enhanced analysis: {category}",
            'original_text': original_message,
            'corrected_text': corrected_message,
            'extracted_keywords': extraction['keywords'],
            'ai_understanding': ai_understanding
        }
        
        # Merge extraction results
        analysis.update(extraction)
        
        # Set primary extracted values
        analysis['extracted_phones'] = extraction['phones']
        analysis['extracted_digits_list'] = extraction['digits']
        analysis['extracted_digits'] = extraction['digits'][0] if extraction['digits'] else None
        analysis['extracted_fin_uin_list'] = extraction['fin_uin']
        analysis['extracted_fin_uin'] = extraction['fin_uin'][0] if extraction['fin_uin'] else None
        analysis['extracted_fan_fcn_list'] = extraction['fan_fcn']
        analysis['extracted_fan_fcn'] = extraction['fan_fcn'][0] if extraction['fan_fcn'] else None
        analysis['extracted_names'] = extraction['names']
        
        return analysis

    def _try_ai_analysis(self, message: str, history: List = None) -> Dict:
        """Try AI analysis with timeout and retry logic"""
        prompt = self._build_analysis_prompt(message, history)
        
        try:
            response = self.client.models.generate_content(
                model="models/gemini-2.5-flash", 
                contents=prompt
            )
            analysis = self._parse_analysis(response.text)
            analysis = self._add_extracted_data(analysis, message)
            return analysis
        except Exception as e:
            # Return error analysis that will trigger fallback
            return {
                'language': 'Unknown',
                'category': 'AI_API_Error',
                'subcategory': 'api_error',
                'confidence': 'low',
                'sentiment': 3,
                'urgency': 'medium',
                'summary': f"AI service temporarily unavailable: {e}",
                'extracted_phones': text_extractor.extract_phone_numbers(message),
                'extracted_digits_list': text_extractor.extract_29_digit_numbers(message),
                'extracted_digits': None,
                'extracted_fin_uin_list': text_extractor.extract_fin_uin_numbers(message),
                'extracted_fin_uin': None,
                'extracted_fan_fcn_list': text_extractor.extract_fan_fcn_numbers(message),
                'extracted_fan_fcn': None,
                'extracted_names': text_extractor.extract_names(message)
            }
    
    def _detect_category(self, text: str, language: str) -> str:
        """Enhanced category detection with new categories and spelling tolerance"""
        
        # NEW CATEGORIES ADDED HERE
        category_patterns = {
            'Greeting': {
                'english': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
                'amharic': ['ሰላም', 'ጤና ይስጥልኝ', 'ደህና መጣህ'],
                'afan_oromo': ['akkam', 'nagaa', 'asim']
            },
            "OTP": {
                "english": [ "otp", "one time password", "security code", "auth code"],
                "amharic": [ "ኦቲፒ", "የደህንነት ኮድ"],
                "afan_oromo": ["otp", "koodii nageenya", "lakkoofsa yeroo tokkotti"]
            },
            'SMS_Resend_Confirmation': {
                'english': ['sms', 'resend', 'confirmation', 'fan', 'fin', 'uin', 'fcn', 'message', 'text', 'code'],
                'amharic': ['ስምስ', 'መልእክት', 'እንደገና', 'ላክ', 'ማረጋገጫ'],
                'afan_oromo': ['sms', 'irra deebii', 'mirkaneessaa', 'ergaa']
            },
            # NEW: Data Update Category
            'Data_Update_Request': {
                'english': ['update', 'change', 'modify', 'correct', 'edit', 'wrong data', 'change name', 'change address', 'update information'],
                'amharic': ['አዘምን', 'ለውጥ', 'ለመቀየር','አስተካክል', 'ቀይር', 'ስም ለውጥ', 'አድራሻ ለውጥ'],
                'afan_oromo': ['jijjiirama', 'fooyya', 'sirreessi', 'maqaa jijjiiramaa', 'teessoo jijjiiramaa']
            },
            # NEW: Account Management Category
            'Account_Management': {
                'english': ['account', 'profile', 'manage account', 'account settings', 'delete account', 'deactivate'],
                'amharic': ['አካውንት', 'መለያ', 'አካውንት ማስተካከል', 'መለያ ማጥፋት'],
                'afan_oromo': ['akkaawuntii', 'manaajii akkaawuntii', 'akkaawuntii haqaa', 'akkaawuntii balleessuu']
            },
            # NEW: Payment Issues Category
            'Payment_Issues': {
                'english': ['payment', 'fee', 'pay', 'transaction', 'money', 'charge', 'refund', 'payment problem'],
                'amharic': ['ክፍያ', 'ገንዘብ', 'ትራንዝአክሽን', 'ሪፈንድ', 'ክፍያ ችግር'],
                'afan_oromo': ['kaffaltii', 'maalessa', 'fe\'ii', 'kaffaltii dhiibaa']
            },
            # NEW: Technical Support Category
            'Technical_Support': {
                'english': ['technical', 'support', 'help desk', 'customer service', 'assistance', 'contact support'],
                'amharic': ['ቴክኒካል', 'ድጋፍ', 'የደንበኞች አገልግሎት', 'እርዳታ'],
                'afan_oromo': ['teknikaa', 'deeggarsaa', 'tajaajila maamilaa', 'gargaarsa']
            },
            'Card_Printing': {
                'english': ['card', 'print', 'printing', 'id card', 'digital id', 'status'],
                'amharic': ['ካርድ', 'ማተም', 'መታተም', 'የካርድ ሁኔታ'],
                'afan_oromo': ['kaardaa', 'maxxansaa', 'haala kaardaa']
            },
            'Demographic_Editing': {
                'english': ['change', 'update', 'edit', 'name', 'address', 'correct', 'modify'],
                'amharic': ['ስም', 'አድራሻ', 'መረጃ', 'አስተካክል','ለመቀየር'],
                'afan_oromo': ['maqaa', 'teessoo', 'odeeffannoo', 'fooyya']
            },
            'Technical_Issues': {
                'english': ['website', 'app', 'error', 'not working', 'technical', 'down', 'crash'],
                'amharic': ['ድርጣቢያ', 'አፕሊኬሽን', 'ስርአት', 'ስህተት', 'አይሰራም'],
                'afan_oromo': ['website', 'app', 'dhiibaa', 'sistemi', 'hin hojjanne']
            },
            'Service_Complaint': {
                'english': ['complaint', 'bad service', 'poor', 'unhappy', 'dissatisfied'],
                'amharic': ['ቅሬታ', 'መጥፎ', 'አገልግሎት', 'አለመደሰት'],
                'afan_oromo': ['gaafii', 'hammeessa', 'tajaajila', 'hin gammadne']
            },
            'Information_Request': {
                'english': ['information', 'office hours', 'requirements', 'fees', 'how', 'what'],
                'amharic': ['መረጃ', 'ቢሮ', 'ሰዓት', 'ምን', 'ያስፈልጋል'],
                'afan_oromo': ['odeeffannoo', 'mana hojii', 'yeroo', 'maal', 'barbaachisa']
            },
            'Foreign_Registration': {
                'english': ['foreign', 'passport', 'international', 'non-citizen'],
                'amharic': ['ውጪ', 'ዜጋ', 'ፓስፖርት'],
                'afan_oromo': ['alaa', 'ummat', 'paasportii']
            },
            'Multiple_Registration': {
                'english': ['duplicate', 'multiple', 'double', 'already registered'],
                'amharic': ['ድርብ', 'ምዝገባ', 'በርካታ'],
                'afan_oromo': ['dachaa', 'galmeessaa', 'baayee']
            },
            'Registration_Process': {
                'english': ['register', 'registration', 'how to', 'process', 'sign up'],
                'amharic': ['ምዝገባ', 'እንዴት', 'ደረጃዎች'],
                'afan_oromo': ['galmeessaa', 'akkamitti', 'raawwii']
            }
        }
        
        # Score categories based on keyword matches
        category_scores = {}
        
        for category, patterns in category_patterns.items():
            score = 0
            lang_patterns = patterns.get(language, []) + patterns.get('english', [])
            
            for pattern in lang_patterns:
                if pattern in text:
                    score += 2  # Exact match
                elif any(self.spelling_tolerance.fuzzy_match(word, pattern) 
                        for word in text.split()):
                    score += 1  # Fuzzy match
            
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'Others'
    
    def _detect_subcategory(self, category: str, text: str) -> str:
        """Detect subcategories for more precise responses"""
        subcategories = {
            'Data_Update_Request': {
                'name_change': ['name', 'ስም', 'maqaa'],
                'address_change': ['address', 'አድራሻ', 'teessoo'],
                'phone_change': ['phone', 'ስልክ', 'bilbila'],
                'email_change': ['email', 'ኢሜል', 'e-mail'],
                'general_update': ['update', 'change', 'ለውጥ', 'jijjiirama']
            },
            'Account_Management': {
                'account_deletion': ['delete', 'remove', 'መጥፋት', 'balleessuu'],
                'account_recovery': ['recover', 'forgot', 'አስመልስ', 'deebisii'],
                'profile_update': ['profile', 'settings', 'መለያ', 'akkaawuntii'],
                'security_settings': ['password', 'security', 'የይለፍ ቃል', 'password']
            },
            'Payment_Issues': {
                'payment_failed': ['failed', 'unsuccessful', 'አልተሳካም', 'hin guutne'],
                'refund_request': ['refund', 'money back', 'ገንዘብ መመለስ', 'deebisii'],
                'payment_method': ['method', 'how to pay', 'ክፍያ ዘዴ', 'karaa kaffaltii'],
                'fee_inquiry': ['fee', 'cost', 'price', 'ክፍያ', 'kaffaltii']
            }
        }
        
        if category in subcategories:
            for subcat, keywords in subcategories[category].items():
                if any(keyword in text for keyword in keywords):
                    return subcat
        
        return 'general'
    
    def _detect_sentiment(self, text: str) -> int:
        """Enhanced sentiment detection"""
        positive_words = ['good', 'great', 'excellent', 'thanks', 'thank you', 'አመሰግናለሁ', 'galatoomaa']
        negative_words = ['bad', 'terrible', 'awful', 'angry', 'frustrated', 'ተናደደ', 'aarsaa']
        
        text_lower = text.lower()
        score = 3  # Neutral
        
        if any(word in text_lower for word in positive_words):
            score += 1
        if any(word in text_lower for word in negative_words):
            score -= 1
        
        return max(1, min(5, score))
    
    def _detect_urgency(self, text: str) -> str:
        """Enhanced urgency detection"""
        urgent_words = ['urgent', 'emergency', 'asap', 'immediately', 'now', 'በቅጽበት', 'yaalaa']
        high_urgency_words = ['important', 'soon', 'need help', 'በጣም አስፈላጊ', 'barbaachisaa']
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in urgent_words):
            return 'high'
        elif any(word in text_lower for word in high_urgency_words):
            return 'medium'
        else:
            return 'low'

    def _build_analysis_prompt(self, message: str, history: List = None) -> str:
        history_context = ""
        if history:
            history_context = "\nPrevious conversation:\n" + "\n".join(
                [f"User: {h.get('user_input', '')}" for h in history[-3:]]
            )
        
        return f"""
        Analyze this customer service message and classify it precisely.

        {history_context}

        Current Message: "{message}"

        CATEGORIES:
        - Greeting: hello, hi, good morning, etc.
        - OTP: otp, otp not woring, etc.
        - SMS_Resend_Confirmation: SMS not received, resend SMS, confirmation message, FAN, FIN, UIN, FCN codes
        - Data_Update_Request: update information, change name, change address, modify data
        - Account_Management: manage account, delete account, profile settings, account recovery
        - Payment_Issues: payment problems, transaction issues, refund requests, fee inquiries
        - Technical_Support: technical help, customer support, assistance needed
        - Demographic_Editing: change name, update address, correct information, edit details
        - Card_Printing: card status, printing delay, collect card, card printing process
        - Technical_Issues: website down, app not working, system error, login problems
        - Service_Complaint: bad service, complaint, poor treatment, unhappy with service
        - Information_Request: office hours, requirements, fees, process information
        - Foreign_Registration: foreigner registration, passport, international, non-citizen
        - Multiple_Registration: duplicate registration, multiple accounts, double record
        - Registration_Process: how to register, registration steps, new registration
        - Others: anything else

        SUBCATEGORIES for Card_Printing:
        - card_status: checking card printing status
        - printing_process: how to print card, card printing steps
        - payment_issue: payment problems for card
        - delivery_issue: card delivery problems

        Provide analysis in this exact format:
        LANGUAGE: [English/Amharic/Afan Oromo/Tigrigna]
        CATEGORY: [exact category from above]
        SUBCATEGORY: [specific subcategory]
        CONFIDENCE: [high/medium/low]
        SENTIMENT: [1-5]
        URGENCY: [high/medium/low]
        SUMMARY: [brief summary]
        """
    
    def _parse_analysis(self, text: str) -> Dict:
        analysis = {
            'language': 'English',
            'category': 'Others',
            'subcategory': 'general',
            'confidence': 'medium',
            'sentiment': 3,
            'urgency': 'medium',
            'summary': 'No summary provided'
        }
        
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("LANGUAGE:"): 
                analysis['language'] = line.split(":", 1)[1].strip()
            elif line.startswith("CATEGORY:"): 
                analysis['category'] = line.split(":", 1)[1].strip()
            elif line.startswith("SUBCATEGORY:"): 
                analysis['subcategory'] = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"): 
                analysis['confidence'] = line.split(":", 1)[1].strip()
            elif line.startswith("SENTIMENT:"): 
                try:
                    analysis['sentiment'] = int(line.split(":", 1)[1].strip())
                except:
                    analysis['sentiment'] = 3
            elif line.startswith("URGENCY:"): 
                analysis['urgency'] = line.split(":", 1)[1].strip()
            elif line.startswith("SUMMARY:"): 
                analysis['summary'] = line.split(":", 1)[1].strip()
        
        return analysis
    
    def _add_extracted_data(self, analysis: Dict, message: str) -> Dict:
        # Enhanced entity extraction
        analysis['extracted_phones'] = text_extractor.extract_phone_numbers(message)
        analysis['extracted_digits_list'] = text_extractor.extract_29_digit_numbers(message)
        analysis['extracted_digits'] = analysis['extracted_digits_list'][0] if analysis['extracted_digits_list'] else None
        analysis['extracted_fin_uin_list'] = text_extractor.extract_fin_uin_numbers(message)
        analysis['extracted_fin_uin'] = analysis['extracted_fin_uin_list'][0] if analysis['extracted_fin_uin_list'] else None
        analysis['extracted_fan_fcn_list'] = text_extractor.extract_fan_fcn_numbers(message)
        analysis['extracted_fan_fcn'] = analysis['extracted_fan_fcn_list'][0] if analysis['extracted_fan_fcn_list'] else None
        analysis['extracted_names'] = text_extractor.extract_names(message)
        
        return analysis

    def _detect_language_fallback(self, text: str) -> str:
        """Enhanced language detection with misspelling tolerance"""
        text_lower = text.lower()
        
        # Check for Amharic characters
        if any(ord(char) >= 0x1200 and ord(char) <= 0x137F for char in text):
            return 'amharic'
        
        # Check for common Amharic words (with tolerance)
        amharic_indicators = ['እባክ', 'ይህ', 'አይ', 'እንደ', 'ሰላም', 'ጤና', 'አመሰግናለሁ']
        if any(indicator in text for indicator in amharic_indicators):
            return 'amharic'
        
        # Check for Afan Oromo words
        oromo_indicators = ['galatoomaa', 'akkam', 'nagaa', 'maal', 'maaloo']
        if any(indicator in text_lower for indicator in oromo_indicators):
            return 'afan_oromo'
        
        return 'english'

# INTELIGENT RESPONSE GENERATOR
class ResponseGenerator:
    """
    Generates responses based on grievance analysis using templates and Gemini API.
    """
    def __init__(self, client=None):
        """
        Initialize the generator.
        
        Args:
            client: The Gemini API client instance (optional).
        """
        self.client = client
        self.response_templates = self._load_templates()

    def _generate_ai_response(self, template: str, analysis: Dict, context: Dict, language: str) -> str:
        """Generate a natural response using AI with the template as policy"""
        if not self.client:
            return None
            
        try:
            # Extract context variables
            user_name = "Customer"
            if analysis.get('extracted_names'):
                user_name = analysis['extracted_names'][0]
            
            # Build prompt
            prompt = f"""
            You are a helpful customer service agent for enterprise.
            
            TASK: Rewrite the following OFFICIAL POLICY into a natural, polite, and helpful response for the user.
            
            OFFICIAL POLICY:
            "{template}"
            
            USER CONTEXT:
            - Name: {user_name}
            - Language: {language}
            - Issue Category: {analysis.get('category')}
            - Subcategory: {analysis.get('subcategory')}
            - Extracted Info: {analysis.get('extracted_phones', []) + analysis.get('extracted_digits_list', [])}
            
            GUIDELINES:
            1. Be polite and professional.
            2. Use the user's name if available.
            3. Keep the core information from the policy exactly as is (URLs, phone numbers, steps).
            4. Adapt the tone to be empathetic if the user is frustrated (Sentiment: {analysis.get('sentiment')}/5).
            5. Respond in {language} language.
            6. Do not make up new procedures. Only use the provided policy.
            
            RESPONSE:
            """
            
            response = self.client.models.generate_content(
                model="models/gemini-2.5-flash", 
                contents=prompt
            )
            return response.text.strip()
            
        except Exception as e:
            print(f"AI response generation failed: {e}")
            return None
    
    def _load_templates(self) -> Dict:
        """Load response templates from JSON file"""
        try:
            file_path = os.path.join('data', 'response_templates.json')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"Template file not found at {file_path}")
                return {}
        except Exception as e:
            print(f"error loading templates: {e}")
            return {}
    
    def generate_response(self, analysis: Dict, context: Dict = None) -> str:
        """Generate intelligent response in the user's language"""
        try:
            category = analysis['category']
            
            # FIXED: Proper language detection with fallback
            detected_lang = analysis.get('language', 'english')
            
            # Handle different language formats
            if isinstance(detected_lang, dict):
                language = detected_lang.get('language', 'english')
            else:
                language = str(detected_lang).lower()
            
            # Map to supported languages
            if any(lang in language for lang in ['amharic', 'amhara', 'ethiopian']):
                response_language = 'amharic'
            elif any(lang in language for lang in ['oromo', 'afan', 'afaan', 'oromia']):
                response_language = 'afan_oromo'
            else:
                response_language = 'english'
            
            # Use AI-corrected understanding if available
            if analysis.get('ai_understanding'):
                ai_info = analysis['ai_understanding']
                
                # If AI has high confidence but we categorized as 'Others', try to use AI intent
                if (ai_info['confidence'] == 'high' and 
                    category == 'Others' and 
                    ai_info['intent'] != 'other'):
                    
                    potential_category = self._map_intent_to_category(ai_info['intent'])
                    if potential_category in self.response_templates:
                        category = potential_category
                        print(f"Using AI-suggested category: {category}")
            
            # Handle special categories
            if category == 'SMS_Resend_Confirmation':
                return self._handle_sms_resend(analysis, response_language)
            elif category == 'Card_Printing':
                return self._handle_card_printing(analysis, response_language)
            
            # For regular categories, use template in detected language
            if category in self.response_templates:
                template = self.response_templates[category]
                
                # Try detected language first
                if response_language in template:
                    response_template = template[response_language]
                    if "PLACEHOLDER" not in response_template:
                        # Try AI generation first
                        ai_response = self._generate_ai_response(response_template, analysis, context, response_language)
                        if ai_response:
                            return ai_response
                        return self._format_response(response_template, analysis, response_language)
                
                # Fallback to English
                if 'english' in template and "PLACEHOLDER" not in template['english']:
                    response_template = template['english']
                    # Try AI generation first
                    ai_response = self._generate_ai_response(response_template, analysis, context, response_language)
                    if ai_response:
                        return ai_response
                    return self._format_response(response_template, analysis, response_language)
            
            # Ultimate fallback
            return self.response_templates['Others'][response_language]
        
        except Exception as e:
            print(f"Error generating response: {e}")
            # Emergency fallback response in detected language
            emergency_response = {
                'english': "I apologize for the technical issue. Please try rephrasing your question or contact our support at +251-11-123-4567.",
                'amharic': "ስለቴክኒካል ችግር እጠይቃለሁ። እባክዎ ጥያቄዎን እንደገና ይግለጹ ወይም በ+251-11-123-4567 ድጋፋችንን ያግኙ።",
                'afan_oromo': "Dhiibaa teknikaa irratti dhiifama gaafadha. Maaloo gaaffii keessan irra deebi'aa ykn deeggarsa keenya +251-11-123-4567 bilbilaa."
            }
            return emergency_response.get(response_language, emergency_response['english'])
    
    def _handle_sms_resend(self, analysis: Dict, language: str) -> str:
        """Handle SMS resend confirmation with enhanced logic"""
        phones = analysis.get('extracted_phones', [])
        digits = analysis.get('extracted_digits_list', [])
        fin_uin = analysis.get('extracted_fin_uin_list', [])
        fan_fcn = analysis.get('extracted_fan_fcn_list', [])
        
        base_templates = {
            'english': {
                "no_info": "Dear Customer,\n\nThank you for reaching out to us.\n\nPlease send us your phone number and the 29-digit registration number from the paper you received when you completed your registration.\n\nWith regards.",
                'has_digits': "Dear Customer,\n\nThank you for reaching out to us.\n\nPlease send us your phone number from the paper you received when you completed your registration.\n\nWith regards.",
                'has_phone': "Dear Customer,\n\nThank you for reaching out to us.\n\nPlease send us your 29-digit registration number from the paper you received when you completed your registration.\n\nWith regards.",
                'has_both': "Enterprise API based on result, this we will give RESPONSE",
                'has_fin_uin': "Found your FIN/UIN record. SMS has been resent to your registered phone.",
                'has_fan_fcn': "Found your FAN/FCN record. SMS has been resent to your registered phone.",
                'auto_detected': "I see you've provided your {entity_type}. I'm resending your SMS confirmation now..."
            },
            'amharic': {
                "no_info": "ጤና ይስጥልን።\n\nእባክዎ ስልክ ቁጥርዎን እና የ29 አሃዝ የመመዝገቢያ ቁጥሩን ምዝገባዎን ሲጨርሱ ከተቀበሉት ወረቀት ላይ ይላኩልን።\n\nእናመሰግናለን።",
                'has_phone': "ጤና ይስጥልን።\n\nእባክዎ የ29 አሃዝ የመመዝገቢያ ቁጥሩን ምዝገባዎን ሲጨርሱ ከተቀበሉት ወረቀት ላይ ይላኩልን።\n\nእናመሰግናለን።",
                'has_digits': "ጤና ይስጥልን።\n\nእባክዎ ስልክ ቁጥርዎን ምዝገባዎን ሲጨርሱ ከተቀበሉት ወረቀት ላይ ይላኩልን።\n\nእናመሰግናለን።",
                'has_both': "SMS ማረጋገጫ ወደ የተመዘገበው ስልክ ቁጥርዎ በድጋሚ ተልኳል። በቅርብ ጊዜ ይገኛል።",
                'has_fin_uin': "FIN/UIN መዝገብዎ ተገኝቷል። SMS ወደ የተመዘገበው ስልክዎ ተልኳል።",
                'has_fan_fcn': "FAN/FCN መዝገብዎ ተገኝቷል። SMS ወደ የተመዘገበው ስልክዎ ተልኳል።",
                'auto_detected': "{entity_type} አቀርብተዋል። የSMS ማረጋገጫዎን አሁን እየላክሁ ነው..."
            },
            'afan_oromo': {
                "no_info": "Kabajamoo Maamila,\n\nNu qunnamaa keessaniif galatoomaa.\n\nMaaloo lakkoofsa bilbila keessan fi lakkoofsa galmee baatii 29 qabu, galmee keessan xumurtan yeroo fudhattan san irraa nuu ergaa.\n\nKabajaan.",
                "has_digits": "Kabajamoo Maamila,\n\nNu qunnamaa keessaniif galatoomaa.\n\nMaaloo lakkoofsa bilbila keessan, galmee keessan xumurtan yeroo fudhattan san irraa nuu ergaa.\n\nKabajaan.",
                "has_phone": "Kabajamoo Maamila,\n\nNu qunnamaa keessaniif galatoomaa.\n\nMaaloo lakkoofsa galmee baatii 29 qabu, galmee keessan xumurtan yeroo fudhattan san irraa nuu ergaa.\n\nKabajaan.",
                'has_both': " Mirkaneessaa SMS gara lakkoofsa bilbilaa galmeeffame keessanitti irra deebi'ameee erge. Fuula duraa argattu.",
                'has_fin_uin': " Galmee FIN/UIN keessan argameera. SMS gara bilbila keessan galmeeffameetti erge.",
                'has_fan_fcn': " Galmee FAN/FCN keessan argameera. SMS gara bilbila keessan galmeeffameetti erge.",
                'auto_detected': " {entity_type} kennitaniif. Mirkaneessaa SMS keessan amma ergaa jira..."
            }
        }
        
        templates = base_templates.get(language, base_templates['english'])
        
        # Check if this was auto-detected (user sent only numbers)
        is_auto_detected = analysis.get('subcategory') == 'automatic_detection'
        
        if is_auto_detected:
            # Determine what type of entity was provided
            entity_type = ""
            if phones:
                entity_type = "phone number" if language == 'english' else "ስልክ ቁጥር" if language == 'amharic' else "lakkoofsa bilbilaa"
            elif digits:
                entity_type = "29-digit registration number" if language == 'english' else "29 አሃዝ የምዝገባ ቁጥር" if language == 'amharic' else "lakkoofsa galmee baatii 29"
            elif fin_uin:
                entity_type = "FIN/UIN" if language == 'english' else "FIN/UIN" if language == 'amharic' else "FIN/UIN"
            elif fan_fcn:
                entity_type = "FAN/FCN" if language == 'english' else "FAN/FCN" if language == 'amharic' else "FAN/FCN"
            
            response = templates['auto_detected'].format(entity_type=entity_type)
            
            # Add what's still needed
            if not phones:
                response += "\n\n Please also provide your phone number to complete the SMS resend."
            elif not digits and not fin_uin and not fan_fcn:
                response += "\n\n Please also provide your 29-digit registration number."
        
        elif fin_uin:
            response = templates['has_fin_uin']
        elif fan_fcn:
            response = templates['has_fan_fcn']
        elif phones and digits:
            response = templates['has_both']
        elif phones and not digits:
            response = templates['has_phone']
        elif digits and not phones:
            response = templates['has_digits']
        else:
            response = templates['no_info']
        
        return self._format_response(response, analysis, language)
    
    def _handle_card_printing(self, analysis: Dict, language: str) -> str:
        """Handle card printing inquiries with intelligent logic"""
        subcategory = analysis.get('subcategory', '').lower()
        
        # Card printing process
        if 'process' in subcategory or 'how' in subcategory or 'print' in subcategory:
            card_process_templates = {
                'english': "Dear Customer,\n\nThank you for reaching out to us.\n\nTo get your Enterprise_name Digital ID card.\n\nWith regards.",
                'amharic': "ጤና ይስጥልን ውድ ደንበኛ፣\n\nለፋይዳ ዲጂታል መታወቂያ ካርድዎ ለማግኘት፣ እባክዎ ድርጣቢያችንን ይጎብኙ።",
                'afan_oromo': "Jaallatamaa Maamila,\n\nNuuf dhaqxi keessanif galatoomaa."
            }
            response = card_process_templates.get(language, card_process_templates['english'])
        
        # Card status check
        else:
            phones = analysis.get('extracted_phones', [])
            fan_fcn = analysis.get('extracted_fan_fcn_list', [])
            names = analysis.get('extracted_names', [])
            
            status_templates = {
                'english': {
                    "no_info": "Dear Customer,\n\nThank you for reaching out to us.\n\nTo check your card printing status, please send us your full name, phone number, and FAN.\n\nWith regards.",
                    "has_info": "Checking your card printing status...\n\nI found your information. Connecting to our card printing system to get the latest status update."
                },
                'amharic': {
                    "no_info": "ጤና ይስጥልን።\n\nየካርድ ማተም ሁኔታዎን ለመፈተሽ፣ እባክዎን ሙሉ ስሞን፣ ስልክ ቁጥር እና FAN ይላኩልን።\n\nእናመሰግናለን።",
                    "has_info": "የካርድ ማተም ሁኔታዎን እየፈተሽን ነው...\n\nመረጃዎን አግኝቻለሁ። አዲሱን የሁኔታ ማሻሻያ ለማግኘት ከካርድ ማተም ስርአታችን ጋር እገናኛለሁ።"
                },
                'afan_oromo': {
                    "no_info": "Kabajamoo Maamila,\n\nNu qunnamaa keessaniif galatoomaa.\n\nHaala maxxansiinsa kaardii keessan beekuu uchun, maaloo maqaa guutuu keessan, lakkoofsa bilbila keessan fi FAN nuu ergaa.\n\nKabajaan.",
                    "has_info": "Haala maxxansaa kaardaa keessan ilaalaa jirra...\n\nOdeeffannoo keessan argadhe. Haala yeroo ammaa jiru argachuuf sirna maxxansaa kaardaa keenya waliin wal qunnamaa jirra."
                }
            }
            
            templates = status_templates.get(language)
            if not templates:
                templates = status_templates['english']
            
            has_sufficient_info = (phones or fan_fcn) and names
            
            if has_sufficient_info:
                response = templates['has_info']
                if phones:
                    response += f"\n\n Phone: {phones[0]}"
                if fan_fcn:
                    response += f"\nFAN/FCN: {fan_fcn[0]}"
                if names:
                    response += f"\nName: {names[0]}"
            else:
                response = templates['no_info']
        
        return self._format_response(response, analysis, language)
    
    def _format_response(self, response: str, analysis: Dict, language: str) -> str:
        """Enhance response with AI understanding context"""
        enhanced = response
        
        # Add AI correction note if significant correction happened
        if (analysis.get('corrected_text') and 
            analysis.get('original_text') != analysis.get('corrected_text') and
            (analysis.get('ai_understanding') or {}).get('confidence') == 'high'):
            
            correction_notes = {
                'english': "\n\nI noticed some spelling issues but understood your request.",
                'amharic': "\n\n የተወሰኑ የፅሁፍ ስህተቶች ነበሩ ግን ጥያቄዎን ተረድቻለሁ።",
                'afan_oromo': "\n\n Dogoggoraa barreeffamaa argameera garuu gaafii keessan hubadhe."
            }
            
            enhanced += correction_notes.get(language, correction_notes['english'])
        
        # Add extracted information
        return self._append_extracted_info(enhanced, analysis, language)
    
    def _map_intent_to_category(self, intent: str) -> str:
        """Map AI intent to category"""
        intent_map = {
            'sms_issue': 'SMS_Resend_Confirmation',
            'card_status': 'Card_Printing',
            'data_update': 'Data_Update_Request',
            'account_help': 'Account_Management',
            'payment': 'Payment_Issues',
            'technical': 'Technical_Support',
            'information': 'Information_Request',
            'complaint': 'Service_Complaint',
            'registration': 'Registration_Process'
        }
        return intent_map.get(intent, 'Others')

    def _append_extracted_info(self, response: str, analysis: Dict, language: str) -> str:
        """Enhance response with extracted data"""
        enhanced = response
        
        # Add labels based on language
        labels = {
            'english': {
                'phone': "Phone detected",
                'digits': "Registration number", 
                'fin_uin': "FIN/UIN detected",
                'fan_fcn': "FAN/FCN detected",
                'name': "Name detected"
            },
            'amharic': {
                'phone': "የተገኘ ስልክ ቁጥር",
                'digits': "የተገኘ ምዝገባ ቁጥር",
                'fin_uin': "የተገኘ FIN/UIN",
                'fan_fcn': "የተገኘ FAN/FCN",
                'name': "የተገኘ ስም"
            },
            'afan_oromo': {
                'phone': "Lakkoofsa bilbilaa argame",
                'digits': "Lakkoofsa galmeessaa argame", 
                'fin_uin': "FIN/UIN argame",
                'fan_fcn': "FAN/FCN argame",
                'name': "Maqaa argame"
            }
        }
        
        lang_labels = labels.get(language, labels['english'])
        
        # Add extracted information
        if analysis.get('extracted_phones'):
            enhanced += f"\n\n{lang_labels['phone']}: {analysis['extracted_phones'][0]}"
        
        if analysis.get('extracted_digits_list'):
            enhanced += f"\n{lang_labels['digits']}: {analysis['extracted_digits_list'][0]}"
        
        if analysis.get('extracted_fin_uin_list'):
            enhanced += f"\n{lang_labels['fin_uin']}: {analysis['extracted_fin_uin_list'][0]}"
        
        if analysis.get('extracted_fan_fcn_list'):
            enhanced += f"\n{lang_labels['fan_fcn']}: {analysis['extracted_fan_fcn_list'][0]}"
        
        if analysis.get('extracted_names'):
            enhanced += f"\n{lang_labels['name']}: {analysis['extracted_names'][0]}"
        
        return enhanced

# KEYWORD SUGGESTIONS
class KeywordSuggestions:
    def __init__(self):
        self.category_keywords = {
            'sms_resend_confirmation': [
                "sms", "resend", "confirmation", "fan", "fin", "uin", "fcn","0912345678","benefit number",
                "message", "text", "not received", "not delivered", "code",
                "የsms", "መልእክት", "እንደገና", "ላክ", "ማረጋገጫ", "sms አልመጣም","ፋይዳ",
                "ergaa", "irra deebii", "mirkaneessaa", "hin argamne", "sms hin argamne"
            ],
            'data_update_request': [
                "update", "change", "modify", "correct", "edit", "wrong data", 
                "change name", "change address", "update information", "personal data",
                "አዘምን", "ለውጥ", "አስተካክል", "ቀይር", "ስም ለውጥ", "አድራሻ ለውጥ", "መረጃ ማደስ",
                "jijjiirama", "fooyya", "sirreessi", "maqaa jijjiiramaa", "teessoo jijjiiramaa"
            ],
            'account_management': [
                "account", "profile", "manage account", "account settings", "delete account", 
                "deactivate", "recover account", "password reset", "security settings",
                "አካውንት", "መለያ", "አካውንት ማስተካከል", "መለያ ማጥፋት", "የይለፍ ቃል መቀየር",
                "akkaawuntii", "manaajii akkaawuntii", "akkaawuntii haqaa", "akkaawuntii balleessuu"
            ],
            'payment_issues': [
                "payment", "fee", "pay", "transaction", "money", "charge", "refund", 
                "payment problem", "failed payment", "money back",
                "ክፍያ", "ገንዘብ", "ትራንዝአክሽን", "ሪፈንድ", "ክፍያ ችግር", "ያልተሳካ ክፍያ",
                "kaffaltii", "maalessa", "fe'ii", "kaffaltii dhiibaa", "kaffaltii hin guutne"
            ],
            'technical_support': [
                "technical", "support", "help desk", "customer service", "assistance", 
                "contact support", "help needed", "technical help",
                "ቴክኒካል", "ድጋፍ", "የደንበኞች አገልግሎት", "እርዳታ", "ቴክኒካል እርዳታ",
                "teknikaa", "deeggarsaa", "tajaajila maamilaa", "gargaarsa", "deeggarsa teknikaa"
            ],
            'demographic_editing': [
                "change", "update", "edit", "name", "address", "information", "details", "personal", "error",
                "ስም", "አድራሻ", "መረጃ", "አስተካክል", "ለወጥ", "የግል", "ስም ለወጥ",
                "maqaa", "teessoo", "odeeffannoo", "jijjiiramaa", "haaromsaa", "sirreessi"
            ],
            'card_printing': [
                "card", "printing", "status", "delay", "collect", "ready",
                "print", "when", "problem", "id card", "digital id", "pick up",
                "የካርድ", "ማተም", "ሁኔታ", "መታተም", "ካርድ", "ማግኘት", "ካርድ ማተም",
                "kaardaa", "maxxansaa", "haala", "qabachuu", "kaardaa maxxansaa"
            ],
            'technical_issues': [
                "website", "app", "error", "technical", "login", "working",
                "system", "problem", "down", "not loading", "crash", "bug",
                "ድርጣቢያ", "አፕሊኬሽን", "ስርአት", "ስህተት", "ችግር", "አይሰራም", "ወደቀ",
                "website", "app", "dhiibaa", "sistemi", "dogoggoraa", "hin hojjanne"
            ],
            'service_complaint': [
                "complaint", "bad", "poor", "unhappy", "service", "treatment",
                "need to complain", "issue", "dissatisfied", "angry", "frustrated",
                "ቅሬታ", "መጥፎ", "አገልግሎት", "አለመደሰት", "ተናደደ", "የማይደሰት",
                "gaafii", "hammeessa", "tajaajila", "hin gammadne", "aarsaa"
            ],
            'information_request': [
                "information", "office", "hours", "requirements", "fees",
                "process", "documents", "how", "what", "needed", "where", "when",
                "መረጃ", "ቢሮ", "ሰዓት", "ምን", "ያስፈልጋል", "ክፍያ", "ሂደት", "ወቅት",
                "odeeffannoo", "mana hojii", "yeroo", "maal", "barbaachisa", "kaffaltii"
            ],
            'foreign_registration': [
                "foreign", "foreigner", "passport", "international", 
                "non-citizen", "expatriate", "alien", "visa", "diplomat",
                "ውጪ", "ዜጋ", "ፓስፖርት", "የተ.ማ.", "ዲፕሎማት",
                "alaa", "ummat", "paasportii", "galmeessaa", "diplomaatii"
            ],
            'multiple_registration': [
                "duplicate", "multiple", "double", "already", "registered",
                "second", "account", "record", "two accounts", "again",
                "ድርብ", "ምዝገባ", "በርካታ", "አካውንት", "ከፍተኛ", "ድርብ ምዝገባ",
                "dachaa", "galmeessaa", "baayee", "akkaawuntii", "galmeessaa dachaa"
            ],
            'registration_process': [
                "register", "registration", "how", "steps", "new", 
                "process", "guide", "start", "help", "procedure", "sign up",
                "ምዝገባ", "እንዴት", "ደረጃዎች", "አዲስ", "ሂደት", "መመዝገብ", "አገናኝ",
                "galmeessaa", "akkamitti", "raawwii", "haaraa", "eeggannoo"
            ]
        }
    
    def get_suggestions(self, current_input: str = "") -> List[str]:
        """Get relevant keyword suggestions based on current input"""
        current_input = current_input.strip().lower()
        
        # If input is very short or empty, show general categories
        if len(current_input) < 2:
            return [
                "SMS not received • SMS አልመጣም • SMS hin argamne",
                "Change name/address • ስም/አድራሻ ለወጥ • Maqaa/teessoo jijjiiramaa", 
                "Card printing status • የካርድ ማተም ሁኔታ • Haala maxxansaa kaardaa",
                "Website/app problem • ድርጣቢያ/አፕ ችግር • Dhiibaa website/app",
                "Office hours • የቢሮ ሰዓት • Yeroo mana hojii",
                "Registration help • የምዝገባ እርዳታ • Gargaarsa galmeessaa"
            ]
        
        # Find matching categories - MORE AGGRESSIVE MATCHING
        matching_categories = {}
        
        for category, keywords in self.category_keywords.items():
            match_score = 0
            matched_keywords = []
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # More aggressive matching
                if (keyword_lower in current_input or 
                    current_input in keyword_lower or
                    any(word in current_input for word in keyword_lower.split()) or
                    any(word in keyword_lower for word in current_input.split())):
                    match_score += 2  # Higher score for better matches
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)
        
        # Also check for partial matches
            for word in current_input.split():
                for keyword in keywords:
                    if word in keyword.lower() or keyword.lower() in word:
                        match_score += 1
                        if keyword not in matched_keywords:
                            matched_keywords.append(keyword)
        
            if match_score > 0:
                matching_categories[category] = {
                    'score': match_score,
                    'keywords': matched_keywords[:6]  # Take more matches
                }
    
        # If we have matches, return them
        if matching_categories:
            # Sort by match score and get top categories
            sorted_categories = sorted(
                matching_categories.items(), 
                key=lambda x: x[1]['score'], 
                reverse=True
            )[:3]  # Top 3 categories
        
            # Collect suggestions
            suggestions = []
            for category, data in sorted_categories:
                suggestions.extend(data['keywords'])
        
            # Remove duplicates and limit
            unique_suggestions = []
            for suggestion in suggestions:
                if suggestion not in unique_suggestions:
                    unique_suggestions.append(suggestion)
        
            return unique_suggestions[:10]  # Return more suggestions
    
        # If no good matches, check for common patterns
        common_patterns = {
            'fan': ['SMS not received', 'FAN code', 'resend SMS', 'confirmation message'],
            'card': ['card status', 'printing delay', 'collect card', 'digital ID'],
            'change': ['change name', 'update address', 'correct information', 'edit details'],
            'website': ['website down', 'app not working', 'technical issue', 'login problem'],
            'how': ['how to register', 'registration process', 'what documents needed', 'requirements']
        }
    
        for pattern, suggestions_list in common_patterns.items():
            if pattern in current_input:
                return suggestions_list
    
        # Ultimate fallback - return general suggestions
        return self.get_suggestions("")

    def display_suggestions(self, current_input: str = ""):
        """Display formatted keyword suggestions"""
        suggestions = self.get_suggestions(current_input)
        
        if not suggestions:
            return
            
        #print("\n Quick suggestions (you can use these keywords):")
        print("─" * 60)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")
        print("─" * 60)
# CONVERSATION MANAGEMENT
class ConversationManager:
    def __init__(self):
        self.sessions = {}
    
    def get_session_context(self, session_id: str) -> Dict:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM session_context WHERE session_id = ?', 
                (session_id,)
            )
            result = cursor.fetchone()
            # conn.close() handled by context manager
        
        if result:
            # Convert string timestamps back to datetime objects if needed
            last_interaction = result['last_interaction']
            if isinstance(last_interaction, str):
                try:
                    last_interaction = datetime.datetime.fromisoformat(last_interaction)
                except:
                    last_interaction = datetime.datetime.now()
            
            return {
                'session_id': result['session_id'],
                'phone_number': result['phone_number'],
                'last_interaction': last_interaction,
                'conversation_history': json.loads(result['conversation_history']) if result['conversation_history'] else [],
                'sms_attempts': result['sms_attempts'],
                'customer_tier': result['customer_tier'],
                'preferred_language': result['preferred_language'],
                'total_interactions': result['total_interactions'],
                'avg_sentiment': result['avg_sentiment'],
                'escalation_count': result['escalation_count']
            }
        return self._create_new_session(session_id)
    
    def _create_new_session(self, session_id: str) -> Dict:
        return {
            'session_id': session_id,
            'phone_number': None,
            'last_interaction': datetime.datetime.now(),
            'conversation_history': [],
            'sms_attempts': 0,
            'customer_tier': 'standard',
            'preferred_language': None,
            'total_interactions': 0,
            'avg_sentiment': 0.5,
            'escalation_count': 0
        }
    
    def update_session_context(self, session_context: Dict):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Convert datetime to ISO string for storage
            last_interaction = session_context['last_interaction']
            if isinstance(last_interaction, datetime.datetime):
                last_interaction = last_interaction.isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO session_context 
                (session_id, phone_number, last_interaction, conversation_history, sms_attempts, customer_tier, preferred_language, total_interactions, avg_sentiment, escalation_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_context['session_id'],
                session_context['phone_number'],
                last_interaction,
                json.dumps(session_context['conversation_history'][-10:]),  # Keep last 10 messages
                session_context['sms_attempts'],
                session_context['customer_tier'],
                session_context['preferred_language'],
                session_context['total_interactions'],
                session_context['avg_sentiment'],
                session_context['escalation_count']
            ))
            
            conn.commit()
            # conn.close() handled by context manager
    
    def add_interaction(self, session_context: Dict, user_input: str, analysis: Dict, response: str):
        interaction = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user_input': user_input,
            'analysis': analysis,
            'response': response
        }
        
        session_context['conversation_history'].append(interaction)
        session_context['total_interactions'] += 1
        session_context['last_interaction'] = datetime.datetime.now()
        
        # Update average sentiment
        current_avg = session_context['avg_sentiment']
        new_sentiment = analysis.get('sentiment', 3)
        total_ints = session_context['total_interactions']
        session_context['avg_sentiment'] = (current_avg * (total_ints - 1) + new_sentiment) / total_ints
        
        # Update preferred language
        if analysis['language'] != 'Unknown':
            session_context['preferred_language'] = analysis['language']
        
        self.update_session_context(session_context)

# UTILITY FUNCTIONS
def generate_session_id() -> str:
    return f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"

def display_analysis_results(analysis: Dict):
    print("\n" + "="*60)
    print("ADVANCED ANALYSIS RESULTS")
    print("="*60)
    print(f"Language:      {analysis['language']}")
    print(f"Category:      {analysis['category']}")
    print(f"Subcategory:   {analysis.get('subcategory', 'N/A')}")
    print(f"Confidence:    {analysis['confidence']}")
    print(f"Sentiment:     {analysis.get('sentiment', 'N/A')}/5")
    print(f"Urgency:       {analysis.get('urgency', 'medium')}")
    
    # Show AI understanding if available
    if analysis.get('ai_understanding'):
        ai_info = analysis['ai_understanding']
        print(f"AI Intent:     {ai_info['intent']} ({ai_info['confidence']})")
    
    if analysis.get('extracted_phones'):
        print(f"Phones:        {analysis['extracted_phones']} -------> to Enterprise API based on result, this we will give RESPONSE")
    if analysis.get('extracted_digits_list'):
        print(f"29-digit:      {analysis['extracted_digits_list']} -------> to Enterprise API based on result, this we will give RESPONSE")
    if analysis.get('extracted_fin_uin_list'):
        print(f"FIN/UIN:       {analysis['extracted_fin_uin_list']} -------> to Enterprise API based on result, this we will give RESPONSE")
    if analysis.get('extracted_fan_fcn_list'):
        print(f"FAN/FCN:       {analysis['extracted_fan_fcn_list']}-------> to Enterprise API based on result, this we will give RESPONSE")
    if analysis.get('extracted_names'):
        print(f"Names:         {analysis['extracted_names']}")
    
    print(f"Summary:       {analysis.get('summary', 'N/A')}")
    print("="*60)

def log_conversation(session_id: str, phone: str, user_input: str, analysis: Dict, response: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (session_id, phone_number, user_input, detected_language, grievance_type, 
             extracted_phones, extracted_digits, extracted_fin_uin, extracted_fan_fcn, 
             response, confidence, sentiment_score, urgency_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            phone,
            user_input,
            analysis['language'],
            analysis['category'],
            json.dumps(analysis.get('extracted_phones', [])),
            analysis.get('extracted_digits'),
            analysis.get('extracted_fin_uin'),
            analysis.get('extracted_fan_fcn'),
            response,
            analysis['confidence'],
            analysis.get('sentiment', 3),
            analysis.get('urgency', 'medium')
        ))
        
        conn.commit()
        # conn.close() handled by context manager

def show_quick_help():
    print("="*50)
    print("="*50)

def validate_user_input(user_input: str) -> Tuple[bool, str]:
    """Validate input"""
    if len(user_input.strip()) < 3:
        return False, "Input too short. Please provide more details."
    
    if len(user_input.strip()) > 500:
        return False, "Input too long. Please summarize your issue."
    
    # Check for common non-descriptive inputs
    non_descriptive = ['hello', 'hi', 'help', 'problem', 'issue', 'yes', 'no']
    if user_input.lower().strip() in non_descriptive:
        return False, "Please describe your specific issue or question."
    
    return True, "Valid input"

def analyze_with_retry(analyzer, message: str, session_id: str, history: List = None, max_retries: int = 1) -> Dict:
    """Analyze with logic"""
    for attempt in range(max_retries + 1):
        try:
            analysis = analyzer.analyze_grievance(message, session_id, history)
            if analysis and analysis.get('category') != 'AI_API_Error':
                return analysis
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries:
            wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    # All retries failed, use fallback
    print("All retries failed, using fallback analysis")
    return analyzer._fallback_analysis(message, message, 'english')

