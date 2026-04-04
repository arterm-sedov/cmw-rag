#!/usr/bin/env python3
"""Batch cleanup for report-pack cross-references (no LLM)."""
from __future__ import annotations

from pathlib import Path

PACK = Path("/workspace/docs/research/executive-research-technology-transfer/report-pack")


def main() -> None:
    # --- 20260331 executive unified ---
    p = PACK / "20260331-research-executive-unified-ru.md"
    t = p.read_text(encoding="utf-8")
    t = t.replace(
        "отчёта «Сайзинг и экономика (CapEx / OpEx / TCO)»_.",
        "отчёта «Сайзинг и экономика (CapEx / OpEx / TCO)».",
    )
    old_tom = (
        "- TOM, роли, фазы внедрения и контроль качества: "
        "_«[Методология](./20260325-research-report-methodology-main-ru.md#method_target_operating_model)»_."
    )
    new_tom = (
        "- TOM, роли, фазы внедрения и контроль качества: в параграфе "
        "_«[Целевая операционная модель (Target Operating Model)](./20260325-research-report-methodology-main-ru.md#method_target_operating_model)»_ "
        "отчёта «Методология разработки и внедрения ИИ»."
    )
    t = t.replace(old_tom, new_tom)
    old_mx = (
        "- Экономика решения и матрица вариантов: "
        "_«[Сайзинг и экономика](./20260325-research-report-sizing-economics-main-ru.md#sizing_decision_matrix)»_."
    )
    new_mx = (
        "- Экономика решения и матрица вариантов: в параграфе "
        "_«[Матрица принятия решений (РБК 2026)](./20260325-research-report-sizing-economics-main-ru.md#sizing_decision_matrix)»_ "
        "отчёта «Сайзинг и экономика (CapEx / OpEx / TCO)»."
    )
    t = t.replace(old_mx, new_mx)
    old_risk = (
        "- Риски бюджета и меры снижения: "
        "_«[Риски и оптимизация](./20260325-research-report-sizing-economics-main-ru.md#sizing_budget_risks_mitigation)»_."
    )
    new_risk = (
        "- Риски бюджета и меры снижения: в параграфе "
        "_«[Риски бюджета и меры снижения](./20260325-research-report-sizing-economics-main-ru.md#sizing_budget_risks_mitigation)»_ "
        "отчёта «Сайзинг и экономика (CapEx / OpEx / TCO)»."
    )
    t = t.replace(old_risk, new_risk)
    old_b = (
        "- Передача ИС/кода и критерии приёмки: "
        "_«[Отчуждение ИС и кода](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)»_."
    )
    new_b = (
        "- Передача ИС/кода и критерии приёмки: в "
        "_Приложении B «[Отчуждение ИС и кода (KT, IP, лицензии, приёмка)](./20260325-research-appendix-b-ip-code-alienation-ru.md#app_b_pack_overview)»_."
    )
    t = t.replace(old_b, new_b)
    old_pd = (
        "- Утверждены правила телеметрии и ПДн (минимизация, ретенция, доступ): "
        "_«[Персональные данные и содержимое в телеметрии (152-ФЗ)](./20260325-research-appendix-d-security-observability-ru.md#app_d__personal_data_telemetry_152fz)»_.\n"
        "- Формализован периметр до LLM и пакет observability в передаче: "
        "_«[Периметр до LLM: минимизация данных, обезличивание и обратимые подстановки](./20260325-research-appendix-d-security-observability-ru.md#app_d__llm_perimeter_data_minimization)»_, "
        "_«[Пакет отчуждения: что добавить по наблюдаемости](./20260325-research-appendix-d-security-observability-ru.md#app_d__handoff_package_observability)»_."
    )
    new_pd = (
        "- Утверждены правила телеметрии и ПДн (минимизация, ретенция, доступ), периметр до LLM и пакет observability в передаче: "
        "Приложение D, параграфы "
        "_«[Персональные данные и содержимое в телеметрии (152-ФЗ)](./20260325-research-appendix-d-security-observability-ru.md#app_d__personal_data_telemetry_152fz)»_, "
        "_«[Периметр до LLM: минимизация данных, обезличивание и обратимые подстановки](./20260325-research-appendix-d-security-observability-ru.md#app_d__llm_perimeter_data_minimization)»_, "
        "_«[Пакет отчуждения: что добавить по наблюдаемости](./20260325-research-appendix-d-security-observability-ru.md#app_d__handoff_package_observability)»_."
    )
    t = t.replace(old_pd, new_pd)
    p.write_text(t, encoding="utf-8")
    print("updated", p.name)

    # --- sizing economics: table rows ---
    p = PACK / "20260325-research-report-sizing-economics-main-ru.md"
    t = p.read_text(encoding="utf-8")
    old_org = (
        "**eval**/FinOps; см. _«[Стратегия внедрения ИИ и организационная зрелость]"
        "(./20260325-research-report-methodology-main-ru.md#method_ai_strategy_org_maturity)»_."
    )
    new_org = (
        "**eval**/FinOps; см. параграф "
        "_«[Стратегия внедрения ИИ и организационная зрелость]"
        "(./20260325-research-report-methodology-main-ru.md#method_ai_strategy_org_maturity)»_ "
        "отчёта «Методология разработки и внедрения ИИ»."
    )
    t = t.replace(old_org, new_org)
    old_opex = (
        "152-ФЗ — в _Приложении D, параграф «[Персональные данные и содержимое в телеметрии]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__personal_data_telemetry_152fz)»_ "
        "и в _Приложении D, параграф «[Периметр до LLM]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__llm_perimeter_data_minimization)»_."
    )
    new_opex = (
        "152-ФЗ — в Приложении D, параграфы "
        "_«[Персональные данные и содержимое в телеметрии]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__personal_data_telemetry_152fz)»_, "
        "_«[Периметр до LLM]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__llm_perimeter_data_minimization)»_."
    )
    t = t.replace(old_opex, new_opex)
    p.write_text(t, encoding="utf-8")
    print("updated", p.name)

    # --- appendix C ---
    p = PACK / "20260325-research-appendix-c-cmw-existing-work-ru.md"
    t = p.read_text(encoding="utf-8")
    old_c = (
        "ориентиры — в _Приложении D, параграф «[Граница доверия, сеть и среда исполнения агента]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__trust_boundary_agent_environment)»_ "
        "и _Приложении D, параграф «[Модель риска, паттерны среды и минимальный состав платформы]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__risk_model_platform_patterns)»_."
    )
    new_c = (
        "ориентиры — в Приложении D, параграфы "
        "_«[Граница доверия, сеть и среда исполнения агента]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__trust_boundary_agent_environment)»_, "
        "_«[Модель риска, паттерны среды и минимальный состав платформы]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__risk_model_platform_patterns)»_."
    )
    t = t.replace(old_c, new_c)
    p.write_text(t, encoding="utf-8")
    print("updated", p.name)

    # --- methodology ---
    p = PACK / "20260325-research-report-methodology-main-ru.md"
    t = p.read_text(encoding="utf-8")
    old_m = (
        "— см. _Приложении D, параграф «[Периметр до LLM]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__llm_perimeter_data_minimization)»_ "
        "и _Приложении D, параграф «[Персональные данные и содержимое в телеметрии]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__personal_data_telemetry_152fz)»_."
    )
    new_m = (
        "— см. в Приложении D, параграфы "
        "_«[Периметр до LLM]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__llm_perimeter_data_minimization)»_, "
        "_«[Персональные данные и содержимое в телеметрии]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__personal_data_telemetry_152fz)»_."
    )
    t = t.replace(old_m, new_m)
    p.write_text(t, encoding="utf-8")
    print("updated", p.name)

    # --- appendix B long line ---
    p = PACK / "20260325-research-appendix-b-ip-code-alienation-ru.md"
    t = p.read_text(encoding="utf-8")
    old_b2 = (
        "см. также _Приложении D, параграф «[Безопасный MVP контура исполнения за 30 дней, дискуссия по средам и выводы]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__secure_mvp_execution_environment)»_. "
        "Детали и соседние паттерны (долгоживущая dev-среда, регулируемый контур) — в _Приложении D, параграф «[Модель риска, паттерны среды и минимальный состав платформы]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__risk_model_platform_patterns)»_."
    )
    new_b2 = (
        "см. также в Приложении D, параграфы "
        "_«[Безопасный MVP контура исполнения за 30 дней, дискуссия по средам и выводы]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__secure_mvp_execution_environment)»_, "
        "_«[Модель риска, паттерны среды и минимальный состав платформы]"
        "(./20260325-research-appendix-d-security-observability-ru.md#app_d__risk_model_platform_patterns)»_."
    )
    t = t.replace(old_b2, new_b2)
    p.write_text(t, encoding="utf-8")
    print("updated", p.name)


if __name__ == "__main__":
    main()
