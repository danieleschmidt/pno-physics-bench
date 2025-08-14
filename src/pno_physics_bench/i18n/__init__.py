"""
Internationalization and Localization Support for PNO Physics Bench

Provides multi-language support for error messages, documentation, and user interfaces
across different regions and compliance requirements.

Supported Languages:
- English (en) - Default
- Spanish (es) 
- French (fr)
- German (de)
- Japanese (ja)
- Chinese Simplified (zh-CN)
"""

import os
import json
from typing import Dict, Optional, Any
from pathlib import Path


class I18nManager:
    """Manages internationalization for the PNO Physics Bench library."""
    
    def __init__(self, default_locale: str = "en"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations = {}
        self._load_translations()
    
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
    
    def get_translation(self, key: str, locale: Optional[str] = None, **kwargs) -> str:
        """Get translated text for a key."""
        
        target_locale = locale or self.current_locale
        
        # Try target locale
        if target_locale in self.translations:
            translation = self._get_nested_value(self.translations[target_locale], key)
            if translation:
                return translation.format(**kwargs) if kwargs else translation
        
        # Try language part of locale
        lang_part = target_locale.split('-')[0] if '-' in target_locale else target_locale
        if lang_part != target_locale and lang_part in self.translations:
            translation = self._get_nested_value(self.translations[lang_part], key)
            if translation:
                return translation.format(**kwargs) if kwargs else translation
        
        # Fallback to default locale
        if self.default_locale in self.translations:
            translation = self._get_nested_value(self.translations[self.default_locale], key)
            if translation:
                return translation.format(**kwargs) if kwargs else translation
        
        # Final fallback - return the key itself
        return key
    
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


__all__ = [
    "I18nManager",
    "get_text", 
    "set_locale",
    "get_current_locale",
    "get_available_locales"
]