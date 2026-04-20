"""Tests for multi-platform CMW config loading.

TDD: Write tests BEFORE implementation.
"""

import pytest


def test_load_cmw_config_default_returns_primary():
    """Backward compat: default platform is 'primary'."""
    from rag_engine.cmw_platform.config import load_cmw_config

    config = load_cmw_config()
    assert "pipeline" in config
    assert config["pipeline"]["input"]["application"] == "systemSolution"


def test_load_cmw_config_with_explicit_primary():
    """Explicit primary platform loads correct config."""
    from rag_engine.cmw_platform.config import load_cmw_config

    config = load_cmw_config("primary")
    assert "pipeline" in config
    assert config["pipeline"]["input"]["application"] == "systemSolution"


def test_load_cmw_config_secondary_loads_different_file():
    """Secondary platform loads cmw_platform_secondary.yaml."""
    from rag_engine.cmw_platform.config import load_cmw_config

    config = load_cmw_config("secondary")
    assert "pipeline" in config
    assert config["pipeline"]["input"]["application"] == "ArchitectureManagement"


def test_get_input_config_accepts_platform_param():
    """get_input_config returns platform-specific input config."""
    from rag_engine.cmw_platform.config import get_input_config

    primary = get_input_config()
    secondary = get_input_config(platform="secondary")
    assert isinstance(primary, dict)
    assert isinstance(secondary, dict)
    assert primary.get("application") == "systemSolution"
    assert secondary.get("application") == "ArchitectureManagement"


def test_get_output_config_accepts_platform_param():
    """get_output_config returns platform-specific output config."""
    from rag_engine.cmw_platform.config import get_output_config

    primary = get_output_config()
    secondary = get_output_config(platform="secondary")
    assert isinstance(primary, dict)
    assert isinstance(secondary, dict)


def test_get_input_attributes_accepts_platform_param():
    """get_input_attributes returns platform-specific mapping."""
    from rag_engine.cmw_platform.config import get_input_attributes

    primary = get_input_attributes()
    secondary = get_input_attributes(platform="secondary")
    assert isinstance(primary, dict)
    assert isinstance(secondary, dict)
    # Primary has support_case_title -> name
    assert "support_case_title" in primary
    # Secondary has document_file -> Commerpredloshenie
    assert "document_file" in secondary


def test_get_platform_attribute_accepts_platform_param():
    """get_platform_attribute works with platform param."""
    from rag_engine.cmw_platform.config import get_platform_attribute

    # Primary: support_case_title -> name
    primary_attr = get_platform_attribute("support_case_title", platform="primary")
    assert primary_attr == "name"

    # Secondary: document_file -> Commerpredloshenie
    secondary_attr = get_platform_attribute("document_file", platform="secondary")
    assert secondary_attr == "Commerpredloshenie"


def test_get_python_attribute_accepts_platform_param():
    """get_python_attribute works with platform param."""
    from rag_engine.cmw_platform.config import get_python_attribute

    # Primary: name -> support_case_title
    primary_attr = get_python_attribute("name", platform="primary")
    assert primary_attr == "support_case_title"

    # Secondary: Commerpredloshenie -> document_file
    secondary_attr = get_python_attribute("Commerpredloshenie", platform="secondary")
    assert secondary_attr == "document_file"


def test_get_request_template_accepts_platform_param():
    """get_request_template returns platform-specific template."""
    from rag_engine.cmw_platform.config import get_request_template

    primary = get_request_template()
    secondary = get_request_template(platform="secondary")
    assert isinstance(primary, str)
    assert isinstance(secondary, str)
    assert "{support_case_title}" in primary
    assert "{document_file}" in secondary or "document" in secondary.lower()


def test_get_template_config_accepts_platform_param():
    """get_template_config works with platform param."""
    from rag_engine.cmw_platform.config import get_template_config

    # Primary
    primary_tpl = get_template_config("systemSolution", "Requests", platform="primary")
    assert primary_tpl is not None
    assert "attributes" in primary_tpl

    # Secondary
    secondary_tpl = get_template_config("ArchitectureManagement", "Zaprosinarazrabotky", platform="secondary")
    assert secondary_tpl is not None
    assert "attributes" in secondary_tpl


def test_get_attribute_metadata_accepts_platform_param():
    """get_attribute_metadata works with platform param."""
    from rag_engine.cmw_platform.config import get_attribute_metadata

    primary = get_attribute_metadata("systemSolution", "Requests", platform="primary")
    assert isinstance(primary, dict)
    assert "support_case_title" in primary

    secondary = get_attribute_metadata("ArchitectureManagement", "Zaprosinarazrabotky", platform="secondary")
    assert isinstance(secondary, dict)
    assert "document_file" in secondary or "summary" in secondary


def test_load_pipeline_config_accepts_platform_param():
    """load_pipeline_config works with platform param."""
    from rag_engine.cmw_platform.config import load_pipeline_config

    primary = load_pipeline_config()
    secondary = load_pipeline_config(platform="secondary")
    assert isinstance(primary, dict)
    assert isinstance(secondary, dict)
    assert primary.get("input", {}).get("application") == "systemSolution"
    assert secondary.get("input", {}).get("application") == "ArchitectureManagement"
