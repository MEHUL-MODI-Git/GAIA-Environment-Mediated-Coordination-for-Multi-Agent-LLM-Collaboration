"""YAML configuration loader"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from string import Template


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file"""
    with open(path) as f:
        return yaml.safe_load(f)


def load_config(path: Path, base_config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config with optional base config merging

    Args:
        path: Path to main config file
        base_config_path: Path to base config to merge with

    Returns:
        Merged configuration dict
    """
    config = load_yaml(path)

    # Merge with base config if specified
    if base_config_path:
        base_config = load_yaml(base_config_path)
        config = merge_configs(base_config, config)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two config dicts, with override taking precedence"""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def substitute_templates(config: Dict[str, Any], variables: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute ${variable} templates in config strings

    Args:
        config: Configuration dict
        variables: Variable substitutions

    Returns:
        Config with variables substituted
    """

    def substitute_value(value):
        if isinstance(value, str):
            return Template(value).safe_substitute(variables)
        elif isinstance(value, dict):
            return substitute_templates(value, variables)
        elif isinstance(value, list):
            return [substitute_value(v) for v in value]
        else:
            return value

    return {k: substitute_value(v) for k, v in config.items()}


from typing import Optional  # Add this import at the top
