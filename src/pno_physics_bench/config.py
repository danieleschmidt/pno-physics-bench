# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Configuration management with validation and environment integration."""

import os
import yaml
import json
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from omegaconf import OmegaConf, DictConfig
import logging

from .validation import ConfigValidator, SecurityValidator
from .exceptions import ConfigurationError, InvalidConfigError, MissingConfigError


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 4
    modes: int = 20
    output_dim: int = 1
    uncertainty_type: str = "diagonal"
    posterior: str = "variational"
    activation: str = "gelu"
    probabilistic_lift_proj: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate model configuration."""
        config_dict = asdict(self)
        ConfigValidator.validate_model_config(config_dict)


@dataclass 
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    kl_weight: float = 1e-4
    num_mc_samples: int = 5
    gradient_clipping: float = 1.0
    mixed_precision: bool = False
    early_stopping: bool = True
    patience: int = 20
    log_interval: int = 10
    visualize_uncertainty: bool = True
    vis_frequency: int = 10
    num_workers: int = 0
    scheduler: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate training configuration."""
        config_dict = asdict(self)
        ConfigValidator.validate_training_config(config_dict)


@dataclass
class DataConfig:
    """Data configuration parameters."""
    pde_name: str
    resolution: int = 64
    num_samples: int = 1000
    normalize: bool = True
    val_split: float = 0.2
    test_split: float = 0.1
    data_path: Optional[str] = None
    generate_on_demand: bool = True
    cache_size: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate data configuration."""
        config_dict = asdict(self)
        ConfigValidator.validate_data_config(config_dict)


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str = "INFO"
    log_file: Optional[str] = None
    json_format: bool = False
    enable_metrics: bool = True
    metrics_port: int = 8000
    enable_colors: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = "pno_experiment"
    description: str = ""
    seed: int = 42
    output_dir: str = "./outputs"
    model: ModelConfig = field(default_factory=lambda: ModelConfig(input_dim=2))
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(pde_name="navier_stokes_2d"))
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize and validate complete configuration."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate all components
        self.model.validate()
        self.training.validate()
        self.data.validate()


class ConfigManager:
    """Centralized configuration management with environment variable support."""
    
    ENV_PREFIX = "PNO_"
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._config: Optional[ExperimentConfig] = None
        self._env_overrides: Dict[str, Any] = {}
        
        # Load environment overrides
        self._load_env_overrides()
    
    def _load_env_overrides(self) -> None:
        """Load configuration overrides from environment variables."""
        env_mappings = {
            f"{self.ENV_PREFIX}LOG_LEVEL": "logging.level",
            f"{self.ENV_PREFIX}BATCH_SIZE": ("training.batch_size", int),
            f"{self.ENV_PREFIX}LEARNING_RATE": ("training.learning_rate", float),
            f"{self.ENV_PREFIX}EPOCHS": ("training.epochs", int),
            f"{self.ENV_PREFIX}SEED": ("seed", int),
            f"{self.ENV_PREFIX}OUTPUT_DIR": "output_dir",
            f"{self.ENV_PREFIX}RESOLUTION": ("data.resolution", int),
            f"{self.ENV_PREFIX}NUM_SAMPLES": ("data.num_samples", int),
            f"{self.ENV_PREFIX}PDE_NAME": "data.pde_name",
            f"{self.ENV_PREFIX}HIDDEN_DIM": ("model.hidden_dim", int),
            f"{self.ENV_PREFIX}NUM_LAYERS": ("model.num_layers", int),
            f"{self.ENV_PREFIX}MODES": ("model.modes", int),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(config_path, tuple):
                    path, converter = config_path
                    try:
                        value = converter(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid value for {env_var}: {value}, error: {e}")
                        continue
                    config_path = path
                
                self._env_overrides[config_path] = value
                logger.info(f"Environment override: {config_path} = {value}")
    
    def load_config(self, config_path: Optional[str] = None) -> ExperimentConfig:
        """Load configuration from file with environment overrides.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded and validated configuration
        """
        if config_path:
            self.config_path = config_path
        
        if self.config_path and Path(self.config_path).exists():
            config_dict = self._load_config_file(self.config_path)
        else:
            # Use default configuration
            config_dict = {}
            if self.config_path:
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
        
        # Apply environment overrides
        config_dict = self._apply_overrides(config_dict, self._env_overrides)
        
        # Convert to configuration objects
        self._config = self._dict_to_config(config_dict)
        
        logger.info(f"Configuration loaded: {self._config.name}")
        
        return self._config
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise MissingConfigError(
                f"Configuration file not found: {config_path}",
                error_code="CONFIG_FILE_NOT_FOUND"
            )
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Security check for YAML
            if not SecurityValidator.validate_yaml_safe(content):
                raise ConfigurationError(
                    "Configuration file contains potentially unsafe YAML constructs",
                    error_code="UNSAFE_YAML_CONTENT"
                )
            
            if config_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(content)
            elif config_path.suffix == '.json':
                config_dict = json.loads(content)
            else:
                raise InvalidConfigError(
                    f"Unsupported configuration file format: {config_path.suffix}",
                    error_code="UNSUPPORTED_CONFIG_FORMAT"
                )
            
            if not isinstance(config_dict, dict):
                raise InvalidConfigError(
                    "Configuration file must contain a dictionary at root level",
                    error_code="INVALID_CONFIG_STRUCTURE"
                )
            
            return config_dict
            
        except yaml.YAMLError as e:
            raise InvalidConfigError(
                f"Invalid YAML in configuration file: {e}",
                error_code="INVALID_YAML"
            )
        except json.JSONDecodeError as e:
            raise InvalidConfigError(
                f"Invalid JSON in configuration file: {e}",
                error_code="INVALID_JSON"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration file: {e}",
                error_code="CONFIG_LOAD_ERROR"
            )
    
    def _apply_overrides(self, config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration.
        
        Args:
            config_dict: Base configuration dictionary
            overrides: Override values with dotted key paths
            
        Returns:
            Configuration with overrides applied
        """
        result = config_dict.copy()
        
        for key_path, value in overrides.items():
            keys = key_path.split('.')
            current = result
            
            # Navigate to the parent dictionary
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    logger.warning(f"Cannot override {key_path}: {key} is not a dictionary")
                    break
                current = current[key]
            else:
                # Set the value
                current[keys[-1]] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to configuration objects.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig object
        """
        try:
            # Handle nested configurations
            model_dict = config_dict.pop('model', {})
            training_dict = config_dict.pop('training', {})
            data_dict = config_dict.pop('data', {})
            logging_dict = config_dict.pop('logging', {})
            
            # Create configuration objects
            model_config = ModelConfig(**model_dict)
            training_config = TrainingConfig(**training_dict)
            data_config = DataConfig(**data_dict)
            logging_config = LoggingConfig(**logging_dict)
            
            # Create main configuration
            config = ExperimentConfig(
                model=model_config,
                training=training_config,
                data=data_config,
                logging=logging_config,
                **config_dict
            )
            
            return config
            
        except TypeError as e:
            raise InvalidConfigError(
                f"Invalid configuration parameters: {e}",
                error_code="INVALID_CONFIG_PARAMS"
            )
    
    def save_config(self, config: ExperimentConfig, output_path: str) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        
        try:
            if output_path.suffix in ['.yaml', '.yml']:
                with open(output_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif output_path.suffix == '.json':
                with open(output_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            else:
                raise InvalidConfigError(
                    f"Unsupported output format: {output_path.suffix}",
                    error_code="UNSUPPORTED_OUTPUT_FORMAT"
                )
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration: {e}",
                error_code="CONFIG_SAVE_ERROR"
            )
    
    def get_config(self) -> ExperimentConfig:
        """Get current configuration.
        
        Returns:
            Current configuration
        """
        if self._config is None:
            raise ConfigurationError(
                "No configuration loaded. Call load_config() first.",
                error_code="NO_CONFIG_LOADED"
            )
        
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update current configuration.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self._config is None:
            raise ConfigurationError(
                "No configuration loaded. Call load_config() first.",
                error_code="NO_CONFIG_LOADED"
            )
        
        # Convert to dictionary, apply updates, convert back
        config_dict = asdict(self._config)
        config_dict = self._apply_overrides(config_dict, updates)
        self._config = self._dict_to_config(config_dict)
        
        logger.info("Configuration updated")
    
    def create_template(self, template_type: str = "basic") -> Dict[str, Any]:
        """Create configuration template.
        
        Args:
            template_type: Type of template to create
            
        Returns:
            Template configuration dictionary
        """
        if template_type == "basic":
            return {
                "name": "pno_experiment",
                "description": "Basic PNO training experiment",
                "seed": 42,
                "output_dir": "./outputs",
                "data": {
                    "pde_name": "navier_stokes_2d",
                    "resolution": 64,
                    "num_samples": 1000,
                    "normalize": True,
                },
                "model": {
                    "input_dim": 2,
                    "hidden_dim": 64,
                    "num_layers": 4,
                    "modes": 20,
                    "uncertainty_type": "diagonal",
                },
                "training": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "kl_weight": 1e-4,
                    "early_stopping": True,
                },
                "logging": {
                    "level": "INFO",
                    "enable_metrics": True,
                }
            }
        elif template_type == "advanced":
            return {
                "name": "pno_advanced_experiment",
                "description": "Advanced PNO training with comprehensive settings",
                "seed": 42,
                "output_dir": "./outputs",
                "tags": ["uncertainty", "pde", "neural_operator"],
                "data": {
                    "pde_name": "navier_stokes_2d",
                    "resolution": 128,
                    "num_samples": 5000,
                    "normalize": True,
                    "val_split": 0.15,
                    "test_split": 0.15,
                },
                "model": {
                    "input_dim": 3,
                    "hidden_dim": 128,
                    "num_layers": 6,
                    "modes": 32,
                    "uncertainty_type": "diagonal",
                    "activation": "gelu",
                },
                "training": {
                    "epochs": 200,
                    "batch_size": 16,
                    "learning_rate": 5e-4,
                    "weight_decay": 1e-4,
                    "kl_weight": 1e-4,
                    "num_mc_samples": 10,
                    "gradient_clipping": 1.0,
                    "mixed_precision": True,
                    "early_stopping": True,
                    "patience": 30,
                    "scheduler": {
                        "type": "cosine",
                        "T_max": 200,
                    },
                },
                "logging": {
                    "level": "INFO",
                    "json_format": True,
                    "enable_metrics": True,
                    "metrics_port": 8000,
                }
            }
        else:
            raise ValueError(f"Unknown template type: {template_type}")


# Convenience functions
def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    manager = ConfigManager()
    return manager.load_config(config_path)


def create_config_template(template_type: str = "basic", output_path: Optional[str] = None) -> Dict[str, Any]:
    """Create and optionally save configuration template.
    
    Args:
        template_type: Type of template
        output_path: Optional path to save template
        
    Returns:
        Template dictionary
    """
    manager = ConfigManager()
    template = manager.create_template(template_type)
    
    if output_path:
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration template saved to {output_path}")
    
    return template