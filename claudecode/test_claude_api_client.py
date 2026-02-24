#!/usr/bin/env python3
"""
Pytest tests for Claude API client.
"""

from unittest.mock import patch, MagicMock

from claudecode.claude_api_client import ClaudeAPIClient


class TestValidateApiAccess:
    """Test validate_api_access uses self.model instead of a hardcoded value."""

    def test_validate_api_access_uses_instance_model(self):
        """Ensure validate_api_access passes self.model to the API call."""
        custom_model = "claude-sonnet-4-20250514"
        client = ClaudeAPIClient(model=custom_model, api_key="fake-key")

        mock_response = MagicMock()
        with patch.object(client.client.messages, "create", return_value=mock_response) as mock_create:
            success, error = client.validate_api_access()

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == custom_model, (
            f"Expected model '{custom_model}', got '{call_kwargs['model']}'"
        )
        assert success is True
        assert error == ""

    def test_validate_api_access_uses_default_model(self):
        """Ensure validate_api_access uses DEFAULT_CLAUDE_MODEL when no model is specified."""
        from claudecode.constants import DEFAULT_CLAUDE_MODEL

        client = ClaudeAPIClient(api_key="fake-key")

        mock_response = MagicMock()
        with patch.object(client.client.messages, "create", return_value=mock_response) as mock_create:
            success, _ = client.validate_api_access()

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == DEFAULT_CLAUDE_MODEL
        assert success is True
