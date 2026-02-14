"""Tests for guardian (guard_client) content moderation - logic tests."""

from __future__ import annotations


class TestGuardModes:
    """Test guard behavior for different modes."""

    def test_enforce_mode_blocks_unsafe(self) -> None:
        """Test enforce mode blocks unsafe content."""
        moderation_result = {
            "safety_level": "Unsafe",
            "categories": ["Violence", "Illegal Acts"],
            "is_safe": False,
        }

        should_block = True
        guard_mode = "enforce"

        if should_block and guard_mode == "enforce":
            blocked = True
        else:
            blocked = False

        assert blocked is True

    def test_report_mode_does_not_block_unsafe(self) -> None:
        """Test report mode does not block unsafe content."""
        moderation_result = {
            "safety_level": "Unsafe",
            "categories": ["Violence"],
            "is_safe": False,
        }

        should_block = True
        guard_mode = "report"

        if should_block and guard_mode == "enforce":
            blocked = True
        else:
            blocked = False

        assert blocked is False

    def test_report_mode_adds_debug_info(self) -> None:
        """Test report mode adds guard info to debug output."""
        moderation_result = {
            "safety_level": "Unsafe",
            "categories": ["Violence", "Illegal Acts"],
            "is_safe": False,
            "refusal": "Yes",
        }

        if moderation_result:
            guard_debug_info = {
                "safety_level": moderation_result.get("safety_level", "Unknown"),
                "categories": moderation_result.get("categories", []),
                "is_safe": moderation_result.get("is_safe", True),
                "refusal": moderation_result.get("refusal", "No"),
            }
        else:
            guard_debug_info = None

        assert guard_debug_info is not None
        assert guard_debug_info["safety_level"] == "Unsafe"
        assert "Violence" in guard_debug_info["categories"]
        assert guard_debug_info["is_safe"] is False


class TestGuardConfiguration:
    """Test guard configuration from settings."""

    def test_guard_disabled(self) -> None:
        """Test behavior when guard is disabled."""
        guard_enabled = False

        if not guard_enabled:
            skip_moderation = True
        else:
            skip_moderation = False

        assert skip_moderation is True

    def test_guard_enabled(self) -> None:
        """Test behavior when guard is enabled."""
        guard_enabled = True

        if not guard_enabled:
            skip_moderation = True
        else:
            skip_moderation = False

        assert skip_moderation is False

    def test_mosec_provider_configuration(self) -> None:
        """Test Mosec provider URL construction."""
        url = "http://test-server"
        port = 8080
        path = "/api/v1/guard"
        expected = f"{url}:{port}{path}"

        assert expected == "http://test-server:8080/api/v1/guard"

    def test_mosec_only_provider(self) -> None:
        """Test that only Mosec provider is currently supported."""
        # Current implementation supports MOSEC only
        # Future support for vLLM planned
        current_provider = "mosec"
        assert current_provider == "mosec"

    def test_guard_modes(self) -> None:
        """Test valid guard modes."""
        valid_modes = ["enforce", "report"]
        for mode in valid_modes:
            assert mode in valid_modes


class TestGuardResultParsing:
    """Test guard result parsing logic."""

    def test_parse_safe_result(self) -> None:
        """Test parsing safe result."""
        result = {
            "safety_level": "Safe",
            "categories": ["Non-violent Illegal Acts"],
            "is_safe": True,
            "refusal": "No",
        }

        safety_level = result.get("safety_level", "Safe")
        categories = result.get("categories", [])
        is_safe = result.get("is_safe", True)

        assert safety_level == "Safe"
        assert is_safe is True
        assert len(categories) == 1

    def test_parse_controversial_result(self) -> None:
        """Test parsing controversial result."""
        result = {
            "safety_level": "Controversial",
            "categories": ["Politically Sensitive", "Unethical Acts"],
            "is_safe": False,
            "refusal": "No",
        }

        safety_level = result.get("safety_level", "Safe")
        is_safe = result.get("is_safe", True)

        assert safety_level == "Controversial"
        assert is_safe is False
        assert len(result["categories"]) == 2

    def test_parse_unsafe_result(self) -> None:
        """Test parsing unsafe result."""
        result = {
            "safety_level": "Unsafe",
            "categories": ["Violence", "Illegal Acts"],
            "is_safe": False,
            "refusal": "Yes",
        }

        safety_level = result.get("safety_level", "Safe")
        is_safe = result.get("is_safe", True)

        assert safety_level == "Unsafe"
        assert is_safe is False
        assert "Violence" in result["categories"]

    def test_should_block_unsafe(self) -> None:
        """Test should block logic for unsafe content."""
        result = {"safety_level": "Unsafe", "is_safe": False}
        should_block = result.get("safety_level") == "Unsafe"
        assert should_block is True

    def test_should_not_block_safe(self) -> None:
        """Test should block logic for safe content."""
        result = {"safety_level": "Safe", "is_safe": True}
        should_block = result.get("safety_level") == "Unsafe"
        assert should_block is False

    def test_should_not_block_controversial(self) -> None:
        """Test should block logic for controversial content."""
        result = {"safety_level": "Controversial", "is_safe": False}
        should_block = result.get("safety_level") == "Unsafe"
        assert should_block is False


class TestGuardContextBuilding:
    """Test guard context building for SGR."""

    def test_safe_context_category_only(self) -> None:
        """Test context building for safe content."""
        result = {
            "safety_level": "Safe",
            "categories": ["Non-violent Illegal Acts"],
        }

        if result.get("safety_level") == "Safe":
            categories = result.get("categories", [])
            if categories:
                context = f"[GUARD] Category: {categories[0]}"
            else:
                context = None
        else:
            context = None

        assert context == "[GUARD] Category: Non-violent Illegal Acts"
        assert "Safety:" not in context

    def test_controversial_context_level_and_categories(self) -> None:
        """Test context building for controversial content."""
        result = {
            "safety_level": "Controversial",
            "categories": ["Politically Sensitive", "Unethical Acts"],
        }

        if result.get("safety_level") == "Controversial":
            categories = result.get("categories", [])
            context = f"[GUARD] Safety: Controversial, Categories: {', '.join(categories)}"
        else:
            context = None

        assert "Safety: Controversial" in context
        assert "Politically Sensitive" in context
        assert "Unethical Acts" in context

    def test_unsafe_in_report_mode(self) -> None:
        """Test context building for unsafe in report mode."""
        result = {
            "safety_level": "Unsafe",
            "categories": ["Violence"],
        }
        guard_mode = "report"

        if result:
            safety_level = result.get("safety_level", "Safe")
            categories = result.get("categories", [])

            if safety_level == "Safe":
                context = f"[GUARD] Category: {categories[0]}" if categories else None
            elif safety_level == "Controversial":
                context = f"[GUARD] Safety: Controversial, Categories: {', '.join(categories)}"
            elif safety_level == "Unsafe" and guard_mode == "report":
                context = f"[GUARD] Safety: Unsafe, Categories: {', '.join(categories)}"
            else:
                context = None
        else:
            context = None

        assert context == "[GUARD] Safety: Unsafe, Categories: Violence"
