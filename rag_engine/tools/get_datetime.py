"""LangChain 1.0 tool for getting current date and time.

This tool provides the current date and time information to help the agent
answer time-related questions and filter information by date periods.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime, tool
from pydantic import BaseModel, Field

from rag_engine.utils.context_tracker import AgentContext

logger = logging.getLogger(__name__)

# Default timezone (can be configured via settings if needed)
DEFAULT_TIMEZONE = "Europe/Moscow"


class GetDateTimeSchema(BaseModel):
    """
    Schema for getting current date and time information.

    This schema defines the input parameters for the get_current_datetime tool.
    Field descriptions are written for LLM understanding and MCP server compatibility.
    """

    timezone: str | None = Field(
        default=None,
        description="Optional timezone identifier (IANA timezone name, e.g., 'Europe/Moscow', "
        "'UTC', 'America/New_York'). If not specified, uses the system default timezone "
        "(Europe/Moscow). Use this to get time in a specific timezone. "
        "RU: Опциональный идентификатор часового пояса (IANA, например, 'Europe/Moscow', "
        "'UTC', 'America/New_York'). Если не указан, используется системный часовой пояс "
        "(Europe/Moscow). Используйте для получения времени в конкретном часовом поясе.",
    )

    include_weekday: bool = Field(
        default=True,
        description="Whether to include the weekday name in the response (e.g., 'Monday', 'Понедельник'). "
        "Useful for questions about specific days of the week. "
        "RU: Включать ли название дня недели в ответ (например, 'Понедельник', 'Monday'). "
        "Полезно для вопросов о конкретных днях недели.",
    )

    format: str | None = Field(
        default=None,
        description="Optional custom date/time format string. If not specified, returns a structured "
        "format with all date/time components. Examples: 'YYYY-MM-DD' for date only, "
        "'HH:MM:SS' for time only. "
        "RU: Опциональный формат даты/времени. Если не указан, возвращается структурированный "
        "формат со всеми компонентами. Примеры: 'YYYY-MM-DD' для даты, 'HH:MM:SS' для времени.",
    )


@tool("get_current_datetime", args_schema=GetDateTimeSchema)
def get_current_datetime(
    timezone: str | None = None,
    include_weekday: bool = True,
    format: str | None = None,
    runtime: ToolRuntime[AgentContext, None] | None = None,
) -> str:
    """Get the current date and time information.

    This tool provides the current date and time to help answer time-related questions,
    filter information by date periods (e.g., "December release", "last month's articles"),
    and determine temporal context for user queries.

    **Use this tool when you need to get the current date and time information:**
    - User asks about current date, time, day of week, or month
    - User asks about time periods (e.g., "December release", "this month's updates")
    - Need to filter or search information by date ranges, or determine if a date is past/present/future
    - User asks "when" questions requiring current time context

    **Examples:**
    - "получи сведения о декабрьском выпуске" → Get current date to determine if December is past/present/future, then search KB for release notes
    - "Что изменилось в этом месяце?" → Get current month, then search KB for recent documentation updates and feature announcements
    - "Когда был последний релиз?" → Get current date to compare with release dates found in KB articles

    **Timezone handling:**
    - By default, returns time in Europe/Moscow timezone
    - You can specify a different timezone if the user asks for time in a specific location
    - UTC is available for universal time reference

    Args:
        timezone: Optional IANA timezone name (e.g., 'Europe/Moscow', 'UTC')
        include_weekday: Whether to include weekday name in response (default: True)
        format: Optional custom format string (if None, returns structured format)

    Returns:
        JSON string containing structured date/time information. Format:
        {
          "datetime": "2024-12-09T15:30:45+03:00",
          "date": "2024-12-09",
          "time": "15:30:45",
          "year": 2024,
          "month": 12,
          "month_name": "December",
          "month_name_ru": "Декабрь",
          "day": 9,
          "weekday": "Monday",
          "weekday_ru": "Понедельник",
          "hour": 15,
          "minute": 30,
          "second": 45,
          "timezone": "Europe/Moscow",
          "timestamp": 1702121445,
          "iso_format": "2024-12-09T15:30:45+03:00"
        }

        All fields are included unless format is specified (then only formatted string is returned).
        Weekday fields are included only if include_weekday is True.
    """
    try:
        # Determine timezone
        tz_str = timezone or DEFAULT_TIMEZONE
        try:
            tz = ZoneInfo(tz_str)
        except Exception:
            logger.warning("Invalid timezone %s, falling back to %s", tz_str, DEFAULT_TIMEZONE)
            tz = ZoneInfo(DEFAULT_TIMEZONE)
            tz_str = DEFAULT_TIMEZONE

        # Get current datetime in specified timezone
        now = datetime.now(tz)

        # If custom format is specified, return formatted string
        if format:
            # Try to use format as Python strftime format directly
            # If it contains common patterns like YYYY, convert them
            try:
                # First try as-is (might already be valid strftime format)
                formatted = now.strftime(format)
            except (ValueError, TypeError):
                # Convert common format patterns to strftime format
                format_map = {
                    "YYYY": "%Y",
                    "MM": "%m",  # Month (uppercase MM)
                    "DD": "%d",
                    "HH": "%H",
                    "mm": "%M",  # Minutes (lowercase mm)
                    "SS": "%S",
                }
                formatted_str = format
                for key, value in format_map.items():
                    formatted_str = formatted_str.replace(key, value)
                try:
                    formatted = now.strftime(formatted_str)
                except (ValueError, TypeError):
                    # Last resort: return ISO format
                    formatted = now.isoformat()
            return json.dumps({"formatted": formatted, "timezone": tz_str}, ensure_ascii=False)

        # Build structured response
        month_names_en = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        month_names_ru = [
            "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
            "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"
        ]
        weekday_names_en = [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]
        weekday_names_ru = [
            "Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"
        ]

        result = {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "year": now.year,
            "month": now.month,
            "month_name": month_names_en[now.month - 1],
            "month_name_ru": month_names_ru[now.month - 1],
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "timezone": tz_str,
            "timestamp": int(now.timestamp()),
            "iso_format": now.isoformat(),
        }

        # Add weekday if requested
        if include_weekday:
            weekday_index = now.weekday()  # 0 = Monday, 6 = Sunday
            result["weekday"] = weekday_names_en[weekday_index]
            result["weekday_ru"] = weekday_names_ru[weekday_index]

        return json.dumps(result, ensure_ascii=False, separators=(',', ':'))

    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting current datetime: %s", exc, exc_info=True)
        return json.dumps(
            {
                "error": f"Failed to get current datetime: {str(exc)}",
                "datetime": None,
            },
            ensure_ascii=False,
        )

