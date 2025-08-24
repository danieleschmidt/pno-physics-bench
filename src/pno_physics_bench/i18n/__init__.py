# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""
Internationalization and Localization Support for PNO Physics Bench

Provides comprehensive multi-language support for error messages, documentation, 
user interfaces, and cultural adaptations across different regions and 
compliance requirements.

Supported Languages:
- English (en) - Default, LTR
- Spanish (es) - LTR  
- French (fr) - LTR
- German (de) - LTR
- Japanese (ja) - LTR
- Chinese Simplified (zh) - LTR
- Arabic (ar) - RTL (future support)
- Hebrew (he) - RTL (future support)

Features:
- Automatic language detection from environment
- Right-to-left (RTL) language support preparation
- Cultural date/time/number formatting
- Browser locale detection
- Fallback locale chains
"""

import os
import json
import locale
import re
from typing import Dict, Optional, Any, List
from pathlib import Path
from datetime import datetime
from decimal import Decimal


class I18nManager:
    """Manages internationalization for the PNO Physics Bench library."""
    
    # Language metadata
    LANGUAGE_INFO = {
        'en': {'name': 'English', 'direction': 'ltr', 'region': 'US'},
        'es': {'name': 'Español', 'direction': 'ltr', 'region': 'ES'},
        'fr': {'name': 'Français', 'direction': 'ltr', 'region': 'FR'},
        'de': {'name': 'Deutsch', 'direction': 'ltr', 'region': 'DE'},
        'ja': {'name': '日本語', 'direction': 'ltr', 'region': 'JP'},
        'zh': {'name': '中文', 'direction': 'ltr', 'region': 'CN'},
        'ar': {'name': 'العربية', 'direction': 'rtl', 'region': 'SA'},
        'he': {'name': 'עברית', 'direction': 'rtl', 'region': 'IL'}
    }
    
    # RTL languages that require special handling
    RTL_LANGUAGES = {'ar', 'he', 'fa', 'ur'}
    
    def __init__(self, default_locale: str = "en"):
        self.default_locale = default_locale
        self.current_locale = self._detect_system_locale() or default_locale
        self.translations = {}
        self.fallback_chain = self._build_fallback_chain()
        self._load_translations()
    
    def _detect_system_locale(self) -> Optional[str]:
        """Detect system locale from environment variables."""
        
        # Check common environment variables
        for env_var in ['LANG', 'LC_ALL', 'LC_MESSAGES']:
            env_locale = os.environ.get(env_var)
            if env_locale:
                # Parse locale string (e.g., "en_US.UTF-8" -> "en")
                locale_match = re.match(r'^([a-z]{2})', env_locale.lower())
                if locale_match:
                    detected_locale = locale_match.group(1)
                    if detected_locale in self.LANGUAGE_INFO:
                        return detected_locale
        
        # Try Python's locale module
        try:
            system_locale, _ = locale.getdefaultlocale()
            if system_locale:
                locale_match = re.match(r'^([a-z]{2})', system_locale.lower())
                if locale_match:
                    detected_locale = locale_match.group(1)
                    if detected_locale in self.LANGUAGE_INFO:
                        return detected_locale
        except Exception:
            pass
            
        return None
    
    def _build_fallback_chain(self) -> List[str]:
        """Build fallback chain for locales."""
        
        fallback_chain = [self.current_locale]
        
        # Add language part if full locale
        if '-' in self.current_locale:
            lang_part = self.current_locale.split('-')[0]
            if lang_part not in fallback_chain:
                fallback_chain.append(lang_part)
        
        # Add English as final fallback if not already present
        if 'en' not in fallback_chain:
            fallback_chain.append('en')
            
        return fallback_chain
    
    def _load_translations(self):
        """Load all available translations."""
        
        translations_dir = Path(__file__).parent / "locales"
        
        if not translations_dir.exists():
            return
            
        for locale_file in translations_dir.glob("*.json"):
            locale_code = locale_file.stem
            try:
                with open(locale_file, 'r', encoding='utf-8') as f:
                    self.translations[locale_code] = json.load(f)
            except Exception:
                # Silently continue if translation file is invalid
                pass
    
    def set_locale(self, locale: str):
        """Set the current locale."""
        if locale in self.translations or locale == self.default_locale:
            self.current_locale = locale
        else:
            # Fallback to language part if full locale not available
            lang_part = locale.split('-')[0] if '-' in locale else locale
            if lang_part in self.translations:
                self.current_locale = lang_part
        
        # Rebuild fallback chain for new locale
        self.fallback_chain = self._build_fallback_chain()
    
    def get_translation(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Get translated text for a key with fallback chain support."""
        
        target_locale = locale or self.current_locale
        fallback_chain = self._build_fallback_chain_for_locale(target_locale)
        
        # Try each locale in fallback chain
        for fallback_locale in fallback_chain:
            if fallback_locale in self.translations:
                translation = self._get_nested_value(self.translations[fallback_locale], key)
                if translation:
                    return translation.format(**kwargs) if kwargs else translation
        
        # Final fallback - return the key itself
        return key
    
    def _build_fallback_chain_for_locale(self, target_locale: str) -> List[str]:
        """Build fallback chain for a specific locale."""
        
        fallback_chain = [target_locale]
        
        # Add language part if full locale
        if '-' in target_locale:
            lang_part = target_locale.split('-')[0]
            if lang_part not in fallback_chain:
                fallback_chain.append(lang_part)
        
        # Add English as final fallback if not already present
        if 'en' not in fallback_chain:
            fallback_chain.append('en')
            
        return fallback_chain
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[str]:
        """Get nested dictionary value using dot notation."""
        
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current if isinstance(current, str) else None
    
    def get_available_locales(self) -> list:
        """Get list of available locales."""
        return list(self.translations.keys())
    
    def is_rtl(self, locale: Optional[str] = None) -> bool:
        """Check if locale uses right-to-left text direction."""
        
        target_locale = locale or self.current_locale
        lang_part = target_locale.split('-')[0] if '-' in target_locale else target_locale
        return lang_part in self.RTL_LANGUAGES
    
    def get_text_direction(self, locale: Optional[str] = None) -> str:
        """Get text direction for locale (ltr or rtl)."""
        
        target_locale = locale or self.current_locale
        lang_part = target_locale.split('-')[0] if '-' in target_locale else target_locale
        return self.LANGUAGE_INFO.get(lang_part, {}).get('direction', 'ltr')
    
    def get_language_name(self, locale: Optional[str] = None) -> str:
        """Get native language name for locale."""
        
        target_locale = locale or self.current_locale
        lang_part = target_locale.split('-')[0] if '-' in target_locale else target_locale
        return self.LANGUAGE_INFO.get(lang_part, {}).get('name', target_locale)
    
    def format_number(self, number: float, locale: Optional[str] = None) -> str:
        """Format number according to locale conventions."""
        
        target_locale = locale or self.current_locale
        lang_part = target_locale.split('-')[0] if '-' in target_locale else target_locale
        
        # Get format pattern from translations or use defaults
        if lang_part in self.translations:
            format_pattern = self._get_nested_value(self.translations[lang_part], "formats.number")
            if format_pattern:
                try:
                    return self._apply_number_format(number, format_pattern)
                except Exception:
                    pass
        
        # Fallback to default formatting
        return self._format_number_default(number, lang_part)
    
    def format_date(self, date: datetime, locale: Optional[str] = None) -> str:
        """Format date according to locale conventions."""
        
        target_locale = locale or self.current_locale
        lang_part = target_locale.split('-')[0] if '-' in target_locale else target_locale
        
        # Get format pattern from translations or use defaults
        if lang_part in self.translations:
            format_pattern = self._get_nested_value(self.translations[lang_part], "formats.date")
            if format_pattern:
                try:
                    return date.strftime(self._convert_date_format(format_pattern))
                except Exception:
                    pass
        
        # Fallback to default formatting
        return self._format_date_default(date, lang_part)
    
    def format_datetime(self, dt: datetime, locale: Optional[str] = None) -> str:
        """Format datetime according to locale conventions."""
        
        target_locale = locale or self.current_locale
        lang_part = target_locale.split('-')[0] if '-' in target_locale else target_locale
        
        # Get format pattern from translations or use defaults
        if lang_part in self.translations:
            format_pattern = self._get_nested_value(self.translations[lang_part], "formats.datetime")
            if format_pattern:
                try:
                    return dt.strftime(self._convert_date_format(format_pattern))
                except Exception:
                    pass
        
        # Fallback to default formatting
        return self._format_datetime_default(dt, lang_part)
    
    def _apply_number_format(self, number: float, format_pattern: str) -> str:
        """Apply number format pattern."""
        
        # Simple implementation for common patterns
        if format_pattern == "#.###,##":  # German/European style
            return f"{number:,.2f}".replace(',', ' ').replace('.', ',').replace(' ', '.')
        elif format_pattern == "#,###.##":  # US/English style  
            return f"{number:,.2f}"
        elif format_pattern == "#,###":  # Integer formatting
            return f"{int(number):,}"
        else:
            return str(number)
    
    def _format_number_default(self, number: float, lang: str) -> str:
        """Default number formatting by language."""
        
        if lang in ['de', 'fr', 'es']:  # European style
            return f"{number:,.2f}".replace(',', ' ').replace('.', ',').replace(' ', '.')
        else:  # US/English style
            return f"{number:,.2f}"
    
    def _convert_date_format(self, pattern: str) -> str:
        """Convert custom date format to strftime format."""
        
        # Simple conversion for common patterns
        conversions = {
            'yyyy': '%Y', 'MM': '%m', 'dd': '%d',
            'HH': '%H', 'mm': '%M', 'ss': '%S'
        }
        
        result = pattern
        for custom, strftime_code in conversions.items():
            result = result.replace(custom, strftime_code)
        
        return result
    
    def _format_date_default(self, date: datetime, lang: str) -> str:
        """Default date formatting by language."""
        
        if lang == 'de':
            return date.strftime('%d.%m.%Y')
        elif lang in ['ja', 'zh']:
            return date.strftime('%Y年%m月%d日')
        else:  # Default to ISO format
            return date.strftime('%Y-%m-%d')
    
    def _format_datetime_default(self, dt: datetime, lang: str) -> str:
        """Default datetime formatting by language."""
        
        if lang == 'de':
            return dt.strftime('%d.%m.%Y %H:%M:%S')
        elif lang in ['ja', 'zh']:
            return dt.strftime('%Y年%m月%d日 %H:%M:%S')
        else:  # Default to ISO format
            return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    def get_browser_locale(self, accept_language: str) -> Optional[str]:
        """Parse browser Accept-Language header to get preferred locale."""
        
        if not accept_language:
            return None
        
        # Parse Accept-Language header (e.g., "en-US,en;q=0.9,de;q=0.8")
        languages = []
        for lang_def in accept_language.split(','):
            lang_def = lang_def.strip()
            if ';' in lang_def:
                lang, quality = lang_def.split(';', 1)
                try:
                    quality_val = float(quality.split('=')[1])
                except (IndexError, ValueError):
                    quality_val = 1.0
            else:
                lang, quality_val = lang_def, 1.0
            
            languages.append((lang.strip(), quality_val))
        
        # Sort by quality and find best match
        languages.sort(key=lambda x: x[1], reverse=True)
        
        for lang, _ in languages:
            # Try exact match first
            if lang in self.translations:
                return lang
            
            # Try language part
            lang_part = lang.split('-')[0] if '-' in lang else lang
            if lang_part in self.translations:
                return lang_part
        
        return None


# Global instance
_i18n_manager = I18nManager()


def get_text(key: str, locale: Optional[str] = None, **kwargs) -> str:
    """Get translated text (convenience function)."""
    return _i18n_manager.get_translation(key, locale, **kwargs)


def set_locale(locale: str):
    """Set current locale (convenience function)."""
    _i18n_manager.set_locale(locale)


def get_current_locale() -> str:
    """Get current locale."""
    return _i18n_manager.current_locale


def get_available_locales() -> list:
    """Get available locales."""
    return _i18n_manager.get_available_locales()


def is_rtl(locale: Optional[str] = None) -> bool:
    """Check if locale uses right-to-left text direction."""
    return _i18n_manager.is_rtl(locale)


def get_text_direction(locale: Optional[str] = None) -> str:
    """Get text direction for locale."""
    return _i18n_manager.get_text_direction(locale)


def get_language_name(locale: Optional[str] = None) -> str:
    """Get native language name."""
    return _i18n_manager.get_language_name(locale)


def format_number(number: float, locale: Optional[str] = None) -> str:
    """Format number according to locale."""
    return _i18n_manager.format_number(number, locale)


def format_date(date: datetime, locale: Optional[str] = None) -> str:
    """Format date according to locale."""
    return _i18n_manager.format_date(date, locale)


def format_datetime(dt: datetime, locale: Optional[str] = None) -> str:
    """Format datetime according to locale."""
    return _i18n_manager.format_datetime(dt, locale)


def detect_browser_locale(accept_language: str) -> Optional[str]:
    """Detect browser locale from Accept-Language header."""
    return _i18n_manager.get_browser_locale(accept_language)


__all__ = [
    "I18nManager",
    "get_text", 
    "set_locale",
    "get_current_locale",
    "get_available_locales",
    "is_rtl",
    "get_text_direction", 
    "get_language_name",
    "format_number",
    "format_date",
    "format_datetime",
    "detect_browser_locale"
]