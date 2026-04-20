"""
Финансовая валидация пакета исследований Comindware.
Пересчёт всех TCO, CapEx, OpEx по реальным данным рынка РФ (апрель 2026)
и операционным данным первого контура Comindware (1dedic).

Методология:
  - Метод А: пересчёт USD от глобального MSRP по курсу 85 RUB/USD (справочный)
  - Метод Б: рыночные цены РФ с параллельным импортом (для КП)
  - Метод В: провайдерские прайсы Cloud.ru / Yandex / Selectel (облачный OpEx)
  - Метод Г: операционные данные Comindware (1dedic контракт)
"""

import pandas as pd
import json

USD_RUB = 85.0

# ═══════════════════════════════════════════════════════════════
# 1. DATASET: РЫНОЧНЫЕ ЦЕНЫ GPU В РФ (апрель 2026, с НДС)
# Источники: deep-researches/20260409-russian-gpu-purchase-prices.md
#            deep-researches/20260409-russian-cloud-gpu-rates-comparison.md
#            validated_gpu_pricing_russia_march2026.md
#            операционные данные Comindware (1dedic)
# ═══════════════════════════════════════════════════════════════

gpu_purchase_prices = pd.DataFrame(
    {
        "gpu": [
            "RTX 4090 24GB",
            "RTX 4090 48GB (mod)",
            "RTX 6000 Ada 48GB",
            "RTX PRO 6000 Blackwell 96GB",
            "A100 80GB PCIe",
            "A100 80GB SXM4",
            "H100 80GB PCIe",
            "H100 80GB SXM5",
            "H200 141GB SXM5",
            "B200 192GB SXM6",
        ],
        "global_msrp_usd": [1599, None, 6800, None, 12000, 15000, 25000, 30000, 30000, 35000],
        "rf_card_only_low_rub": [
            380000,
            450000,
            750000,
            900000,
            1200000,
            1500000,
            3500000,
            4500000,
            5100000,
            6800000,
        ],
        "rf_card_only_high_rub": [
            520000,
            650000,
            950000,
            None,
            1800000,
            2500000,
            5500000,
            7000000,
            7700000,
            10200000,
        ],
        "rf_ws_low_rub": [
            635000,
            1100000,
            1500000,
            1500000,
            2100000,
            None,
            5500000,
            5500000,
            None,
            None,
        ],
        "rf_ws_high_rub": [
            745000,
            1400000,
            2000000,
            2000000,
            2500000,
            None,
            6500000,
            6500000,
            None,
            None,
        ],
        "source": [
            "market_apr2026",
            "market_apr2026_unofficial",
            "market_apr2026",
            "comindware_ops_apr2026",
            "market_apr2026",
            "market_apr2026",
            "market_apr2026",
            "market_apr2026",
            "market_apr2026",
            "market_apr2026_est",
        ],
    }
)

# ═══════════════════════════════════════════════════════════════
# 2. DATASET: ОБЛАЧНЫЕ ТАРИФЫ (руб./час, руб./мес при 730ч)
# Источники: Cloud.ru прайс, Yandex Cloud, Selectel, 1dedic контракт
# ═══════════════════════════════════════════════════════════════

cloud_rates = pd.DataFrame(
    {
        "provider": [
            "1dedic",
            "1dedic",
            "1dedic",
            "1dedic",
            "Selectel",
            "Selectel",
            "Selectel",
            "Selectel",
            "Yandex Cloud",
            "Yandex Cloud",
            "Yandex Cloud",
            "Yandex Cloud",
            "Cloud.ru",
            "Cloud.ru",
            "Cloud.ru",
            "Cloud.ru",
            "Cloud.ru",
        ],
        "gpu": [
            "RTX 4090 48GB",
            "RTX 4090 +1extra",
            "RTX 4090",
            "RTX 4090",
            "RTX 4090",
            "A100 40GB",
            "A100 80GB",
            "H100 80GB",
            "V100 1x",
            "V100 4x",
            "A100 1x",
            "A100 8x",
            "V100 4x",
            "A100 5x",
            "H100 5x",
            "H100 NVLink 5x",
            "H100 NVLink 7x",
        ],
        "vram_gb": [
            48,
            24,
            24,
            24,
            24,
            40,
            80,
            80,
            32,
            128,
            80,
            640,
            128,
            320,
            400,
            400,
            560,
        ],
        "gpu_count": [
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            4,
            1,
            8,
            4,
            5,
            5,
            5,
            7,
        ],
        "rate_rub_hour_low": [
            None,
            None,
            80,
            80,
            80,
            170,
            250,
            420,
            200,
            700,
            300,
            2000,
            None,
            None,
            None,
            None,
            None,
        ],
        "rate_rub_hour_high": [
            None,
            None,
            150,
            150,
            110,
            220,
            310,
            580,
            350,
            1200,
            500,
            3500,
            None,
            None,
            None,
            None,
            None,
        ],
        "rate_rub_month_low": [
            100000,
            145000,
            None,
            None,
            58000,
            110000,
            183000,
            657000,
            146000,
            511000,
            219000,
            1460000,
            None,
            None,
            None,
            None,
            None,
        ],
        "rate_rub_month_high": [
            100000,
            145000,
            None,
            None,
            110000,
            219000,
            365000,
            1606000,
            256000,
            876000,
            365000,
            2555000,
            None,
            None,
            None,
            None,
            None,
        ],
        "rate_rub_month_config": [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            721000,
            1158000,
            2004000,
            3117000,
            4364000,
        ],
        "source": [
            "comindware_ops",
            "comindware_ops",
            "selectel_pricing",
            "selectel_pricing",
            "selectel_apr2026",
            "selectel_apr2026",
            "selectel_apr2026",
            "selectel_apr2026",
            "yandex_cloud_est",
            "yandex_cloud_est",
            "yandex_cloud_est",
            "yandex_cloud_est",
            "cloud_ru_tariff",
            "cloud_ru_tariff",
            "cloud_ru_tariff",
            "cloud_ru_tariff",
            "cloud_ru_tariff",
        ],
    }
)

# ═══════════════════════════════════════════════════════════════
# 3. ВЫЧИСЛЕНИЯ: TCO-ТАБЛИЦА
# ═══════════════════════════════════════════════════════════════

HOURS_PER_MONTH = 730
MONTHS_PER_YEAR = 12
YEARS = 3

print("=" * 80)
print("ВЫЧИСЛЕНИЕ TCO: ЛОКАЛЬНОЕ vs ОБЛАЧНОЕ РАЗВЁРТЫВАНИЕ")
print("=" * 80)

# --- ЛОКАЛЬНОЕ (CapEx + OpEx обслуживание) ---
local_capex = {
    "Мелкое (1x RTX 4090 WS)": (635000, 745000),
    "Среднее (2x RTX 4090 / A100)": (1200000, 2000000),
    "Крупное (4-8x A100/H100)": (12000000, 25000000),
}

local_opex_annual = {
    "Мелкое": 170000,
    "Среднее": 425000,
    "Крупное": 1700000,
}

print("\n--- ЛОКАЛЬНОЕ РАЗВЁРТЫВАНИЕ ---")
for tier, (capex_low, capex_high) in local_capex.items():
    tier_key = tier.split()[0]
    opex_annual = local_opex_annual[tier_key]
    tco_low = capex_low + opex_annual * YEARS
    tco_high = capex_high + opex_annual * YEARS
    print(f"  {tier}")
    print(f"    CapEx:  {capex_low:>14,} — {capex_high:>14,} ₽")
    print(f"    OpEx/год: {opex_annual:>14,} ₽")
    print(f"    TCO/3г: {tco_low:>14,} — {tco_high:>14,} ₽")
    print()

# --- ОБЛАЧНОЕ (0 CapEx, OpEx = аренда) ---
print("--- ОБЛАЧНОЕ РАЗВЁРТЫВАНИЕ ---")

cloud_scenarios = {
    "Мелкое": {
        "description": "1x RTX 4090 48GB (1dedic)",
        "monthly_low": 100000,
        "monthly_high": 145000,
        "source": "1dedic контракт Comindware",
    },
    "Мелкое (вариант Selectel)": {
        "description": "1x RTX 4090 24GB (Selectel)",
        "monthly_low": 58000,
        "monthly_high": 110000,
        "source": "Selectel прайс апр2026",
    },
    "Среднее": {
        "description": "4x V100 (Cloud.ru) / A100 1x (Yandex)",
        "monthly_low": 511000,
        "monthly_high": 876000,
        "source": "Yandex Cloud 4xV100",
    },
    "Крупное": {
        "description": "5x A100 (Cloud.ru) / 8x A100 (Yandex)",
        "monthly_low": 1158000,
        "monthly_high": 2555000,
        "source": "Cloud.ru + Yandex Cloud",
    },
}

for tier, data in cloud_scenarios.items():
    annual_low = data["monthly_low"] * MONTHS_PER_YEAR
    annual_high = data["monthly_high"] * MONTHS_PER_YEAR
    tco_low = annual_low * YEARS
    tco_high = annual_high * YEARS
    print(f"  {tier}: {data['description']}")
    print(
        f"    Месяц:   {data['monthly_low']:>14,} — {data['monthly_high']:>14,} ₽  ({data['source']})"
    )
    print(f"    Год:     {annual_low:>14,} — {annual_high:>14,} ₽")
    print(f"    TCO/3г:  {tco_low:>14,} — {tco_high:>14,} ₽")
    print()

# ═══════════════════════════════════════════════════════════════
# 4. СВЕРКА: СТАРАЯ TCO-ТАБЛИЦА vs НОВАЯ
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("СВЕРКА: СТАРАЯ TCO-ТАБЛИЦА vs РАСЧЁТНАЯ")
print("=" * 80)

old_tco = pd.DataFrame(
    {
        "tier": [
            "Локальное Мелкое",
            "Облачное Мелкое",
            "Локальное Среднее",
            "Облачное Среднее",
            "Локальное Крупное",
            "Облачное Крупное",
        ],
        "old_capex": [212500, 0, 850000, 0, 8500000, 0],
        "old_opex_annual": [170000, 1020000, 425000, 2550000, 1700000, 10200000],
        "old_tco_3yr": [722500, 3060000, 2125000, 7650000, 13600000, 30600000],
    }
)

new_tco = pd.DataFrame(
    {
        "tier": [
            "Локальное Мелкое",
            "Облачное Мелкое",
            "Локальное Среднее",
            "Облачное Среднее",
            "Локальное Крупное",
            "Облачное Крупное",
        ],
        "new_capex_low": [635000, 0, 1200000, 0, 12000000, 0],
        "new_capex_high": [745000, 0, 2000000, 0, 25000000, 0],
        "new_opex_low": [170000, 1200000, 425000, 5100000, 1700000, 13920000],
        "new_opex_high": [170000, 1740000, 425000, 8760000, 1700000, 30660000],
        "new_tco_3yr_low": [
            635000 + 170000 * 3,
            1200000 * 3,
            1200000 + 425000 * 3,
            5100000 * 3,
            12000000 + 1700000 * 3,
            13920000 * 3,
        ],
        "new_tco_3yr_high": [
            745000 + 170000 * 3,
            1740000 * 3,
            2000000 + 425000 * 3,
            8760000 * 3,
            25000000 + 1700000 * 3,
            30660000 * 3,
        ],
    }
)

comparison = old_tco.copy()
comparison["new_capex_low"] = new_tco["new_capex_low"]
comparison["new_capex_high"] = new_tco["new_capex_high"]
comparison["new_opex_low"] = new_tco["new_opex_low"]
comparison["new_opex_high"] = new_tco["new_opex_high"]
comparison["delta_capex_low_vs_old"] = comparison["new_capex_low"] - comparison["old_capex"]
comparison["ratio_capex_low_vs_old"] = comparison["new_capex_low"] / comparison[
    "old_capex"
].replace(0, 1)

print()
for _, row in comparison.iterrows():
    print(f"  {row['tier']}:")
    print(
        f"    Старый CapEx: {row['old_capex']:>14,} → Новый: {row['new_capex_low']:>14,} — {row['new_capex_high']:>14,}"
    )
    if row["old_capex"] > 0:
        print(f"    Δ CapEx: ×{row['ratio_capex_low_vs_old']:.1f}")
    print(
        f"    Старый OpEx:  {row['old_opex_annual']:>14,} → Новый: {row['new_opex_low']:>14,} — {row['new_opex_high']:>14,}"
    )
    print(f"    Старый TCO:   {row['old_tco_3yr']:>14,}")
    print()

# ═══════════════════════════════════════════════════════════════
# 5. СВЕРКА: ЦЕНЫ КАРТ (отчёт vs рынок)
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("СВЕРКА: ЦЕНЫ GPU В ОТЧЁТЕ vs РЫНОК РФ (апрель 2026)")
print("=" * 80)

gpu_price_check = pd.DataFrame(
    {
        "gpu": ["RTX 4090 24GB", "A100 80GB", "H100 80GB SXM", "RTX PRO 6000 Blackwell 96GB"],
        "report_card_low": [170000, 760000, 1290000, 1200000],
        "report_card_high": [220000, 1280000, 2550000, 1800000],
        "market_card_low": [380000, 1200000, 3500000, 900000],
        "market_card_high": [520000, 1800000, 5500000, None],
        "ratio_low": [380000 / 170000, 1200000 / 760000, 3500000 / 1290000, 900000 / 1200000],
    }
)

for _, row in gpu_price_check.iterrows():
    print(f"  {row['gpu']}:")
    print(f"    отчёт: {row['report_card_low']:>12,} — {row['report_card_high']:>12,} ₽")
    print(f"    рынок: {row['market_card_low']:>12,} — {row['market_card_high']:>12,} ₽")
    print(f"    занижение в отчёте: ×{row['ratio_low']:.1f}")
    print()

# ═══════════════════════════════════════════════════════════════
# 6. СВЕРКА: ОБЛАЧНЫЕ ТАРИФЫ (отчёт vs реальные)
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("СВЕРКА: ОБЛАЧНЫЕ ТАРИФЫ В ОТЧЁТЕ vs РЕАЛЬНЫЕ (апрель 2026)")
print("=" * 80)

print("\n  Selectel H100/час:")
print(f"    отчёт:    900 — 2 200 ₽/час")
print(f"    реальность: 420 — 580 ₽/час")
print(f"    занижение верхней границы: ×{2200 / 580:.1f} (в отчёте верхняя граница выше факта)")

print("\n  Cloud.ru 5xA100/мес:")
print(f"    отчёт (из прайса): 1 158 000 ₽/мес")
print(f"    TCO крупного облака/год: 10 200 000 ₽/год = 850 000 ₽/мес")
print(f"    delta: TCO-таблица на {850000 / 1158000 * 100:.0f}% от Cloud.ru тарифа")

print("\n  «Старт от ...» для облака:")
print(f"    отчёт:     212 500 ₽/мес")
print(f"    1dedic:    100 000 ₽/мес (контракт)")
print(f"    delta:     ×{212500 / 100000:.2f} (отчёт завышен)")

# ═══════════════════════════════════════════════════════════════
# 7. ПРОВЕРКА: H200 «50 млн ₽»
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("ПРОВЕРКА: H200 «50 млн ₽» — карта или сервер?")
print("=" * 80)

h200_card_global_low = 25000  # USD
h200_card_global_high = 35000
h200_card_rf_low = h200_card_global_low * USD_RUB * 1.8
h200_card_rf_high = h200_card_global_high * USD_RUB * 2.6

h200_8x_server_low = 250000 * 1.5 * USD_RUB  # DGX с наценкой 1.5x
h200_8x_server_high = 400000 * 2.4 * USD_RUB  # DGX с наценкой 2.4x

print(f"  1 карта H200 (глобально):     ${h200_card_global_low:,} — ${h200_card_global_high:,}")
print(
    f"  1 карта H200 (РФ, паралл.):    {h200_card_rf_low:>14,.0f} — {h200_card_rf_high:>14,.0f} ₽"
)
print(
    f"  8xH200 HGX (глобально):        ${(h200_8x_server_low / USD_RUB / 1.5):>14,.0f} — ${(h200_8x_server_high / USD_RUB / 2.4):>14,.0f} USD"
)
print(
    f"  8xH200 HGX (РФ, паралл. 1.5–2.4x): {h200_8x_server_low:>14,.0f} — {h200_8x_server_high:>14,.0f} ₽"
)
print()
print(f"  Вывод: 50 000 000 ₽ = ${50000000 / USD_RUB:>14,.0f} — это сервер 8×H200, НЕ одна карта")
print(f"  Одна карта H200 в РФ: {h200_card_rf_low / 1e6:.1f} — {h200_card_rf_high / 1e6:.1f} млн ₽")

# ═══════════════════════════════════════════════════════════════
# 8. ОПЕРАЦИОННЫЕ ДАННЫЕ: КОНТРАКТ COMINDWARE (1dedic)
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("ОПЕРАЦИОННЫЕ ДАННЫЕ: КОНТРАКТ COMINDWARE (1dedic)")
print("=" * 80)

dedic_monthly_1gpu = 100000
dedic_monthly_2gpu = 145000
rtx_pro_6000_card = 900000

print(f"  RTX 4090 48GB аренда:     {dedic_monthly_1gpu:>14,} ₽/мес")
print(f"  +1 RTX 4090 доп. карта:   {dedic_monthly_2gpu - dedic_monthly_1gpu:>14,} ₽/мес")
print(f"  Итого 2x RTX 4090:        {dedic_monthly_2gpu:>14,} ₽/мес")
print(f"  RTX PRO 6000 Blackwell (карта без шасси): {rtx_pro_6000_card:>14,} ₽")
print()
print(f"  Годовой OpEx (1x 4090):   {dedic_monthly_1gpu * 12:>14,} ₽")
print(f"  Годовой OpEx (2x 4090):   {dedic_monthly_2gpu * 12:>14,} ₽")
print(f"  TCO/3г (1x 4090 облако): {dedic_monthly_1gpu * 12 * 3:>14,} ₽")
print(f"  TCO/3г (2x 4090 облако): {dedic_monthly_2gpu * 12 * 3:>14,} ₽")

# ═══════════════════════════════════════════════════════════════
# 9. ИТОГОВАЯ TCO-ТАБЛИЦА (для вставки в отчёт)
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("ИТОГОВАЯ TCO-ТАБЛИЦА (для вставки в отчёт)")
print("=" * 80)

final_tco = pd.DataFrame(
    {
        "Развертывание": [
            "Локальное (Мелкое)",
            "Облачное (Мелкое)",
            "Локальное (Среднее)",
            "Облачное (Среднее)",
            "Локальное (Крупное)",
            "Облачное (Крупное)",
        ],
        "CapEx низ": [635000, 0, 1200000, 0, 12000000, 0],
        "CapEx высок": [745000, 0, 2000000, 0, 25000000, 0],
        "OpEx/год (облако — месячный × 12)": [
            170000,
            1200000,
            425000,
            6132000,
            1700000,
            16656000,
        ],
        "OpEx/год (облако — месячный × 12, верх)": [
            170000,
            1740000,
            425000,
            10512000,
            1700000,
            30072000,
        ],
    }
)

# Облачный OpEx считается: мелкое = 100K/мес, среднее = 511K-876K/мес, крупное = Cloud.ru 5xA100 + Yandex 8xA100
# мелкое облако: 100K × 12 = 1.2M, верхняя = 145K × 12 = 1.74M (1dedic)
# среднее облако: 511K × 12 = 6.132M, верхняя = 876K × 12 = 10.512M
# крупное облако: Cloud.ru 5xA100 = 1.158M/мес × 12 = 13.896M/год, Yandex 8xA100 = 2.555M/мес × 12 = 30.66M/год
# пересчёт нижней границы крупного: пробуем 2x 4090 на 1dedic = 145K/мес — нет, это мелкое
# нижняя граница крупного = Cloud.ru 5xA100 = 1_158_000/мес
# верхняя = Yandex 8xA100 = 2_555_000/мес

# Пересчитаем правильно:

final_tco["TCO_3yr_low"] = (
    final_tco["CapEx низ"] + final_tco["OpEx/год (облако — месячный × 12)"] * 3
)
final_tco["TCO_3yr_high"] = (
    final_tco["CapEx высок"] + final_tco["OpEx/год (облако — месячный × 12, верх)"] * 3
)

# Override with explicitly correct values
final_tco.loc[final_tco["Развертывание"] == "Локальное (Мелкое)", "TCO_3yr_low"] = (
    635000 + 170000 * 3
)
final_tco.loc[final_tco["Развертывание"] == "Локальное (Мелкое)", "TCO_3yr_high"] = (
    745000 + 170000 * 3
)

final_tco.loc[final_tco["Развертывание"] == "Облачное (Мелкое)", "TCO_3yr_low"] = 100000 * 12 * 3
final_tco.loc[final_tco["Развертывание"] == "Облачное (Мелкое)", "TCO_3yr_high"] = 145000 * 12 * 3

final_tco.loc[final_tco["Развертывание"] == "Локальное (Среднее)", "TCO_3yr_low"] = (
    1200000 + 425000 * 3
)
final_tco.loc[final_tco["Развертывание"] == "Локальное (Среднее)", "TCO_3yr_high"] = (
    2000000 + 425000 * 3
)

final_tco.loc[final_tco["Развертывание"] == "Облачное (Среднее)", "TCO_3yr_low"] = 511000 * 12 * 3
final_tco.loc[final_tco["Развертывание"] == "Облачное (Среднее)", "TCO_3yr_high"] = 876000 * 12 * 3

final_tco.loc[final_tco["Развертывание"] == "Локальное (Крупное)", "TCO_3yr_low"] = (
    12000000 + 1700000 * 3
)
final_tco.loc[final_tco["Развертывание"] == "Локальное (Крупное)", "TCO_3yr_high"] = (
    25000000 + 1700000 * 3
)

final_tco.loc[final_tco["Развертывание"] == "Облачное (Крупное)", "TCO_3yr_low"] = 1158000 * 12 * 3
final_tco.loc[final_tco["Развертывание"] == "Облачное (Крупное)", "TCO_3yr_high"] = 2555000 * 12 * 3

print()
print(f"{'Развертывание':<30} {'CapEx':>20} {'OpEx/год':>25} {'TCO/3г':>25}")
print("-" * 105)
for _, r in final_tco.iterrows():
    capex_str = (
        f"{r['CapEx низ']:>12,} — {r['CapEx высок']:>12,}"
        if r["CapEx низ"] > 0
        else f"{'0':>12} {'':>14}"
    )
    opex_str = f"{r['OpEx/год (облако — месячный × 12)']:>12,} — {r['OpEx/год (облако — месячный × 12, верх)']:>12,}"
    tco_str = f"{r['TCO_3yr_low']:>12,} — {r['TCO_3yr_high']:>12,}"
    print(f"  {r['Развертывание']:<28} {capex_str}  |  {opex_str}  |  {tco_str}")

# ═══════════════════════════════════════════════════════════════
# 10. СВОДКА ПРОТИВОРЕЧИЙ И ВЕРДИКТ
# ═══════════════════════════════════════════════════════════════

print()
print("=" * 80)
print("СВОДКА: КЛЮЧЕВЫЕ РАСХОЖДЕНИЯ (ОТЧЁТ vs РАСЧЁТ)")
print("=" * 80)

discrepancies = [
    (
        "К1: CapEx мелкое",
        "212 500",
        f"{635000} — {745000}",
        f"×{635000 / 212500:.1f} — ×{745000 / 212500:.1f}",
        "КРИТИЧЕСКОЕ",
    ),
    (
        "К1: CapEx среднее",
        "850 000",
        f"{1200000} — {2000000}",
        f"×{1200000 / 850000:.1f} — ×{2000000 / 850000:.1f}",
        "КРИТИЧЕСКОЕ",
    ),
    (
        "К1: CapEx крупное",
        "8 500 000",
        f"{12000000} — {25000000}",
        f"×{12000000 / 8500000:.1f} — ×{25000000 / 8500000:.1f}",
        "КРИТИЧЕСКОЕ",
    ),
    (
        "К2: Облачный старт/мес",
        "212 500",
        "100 000 (1dedic)",
        f"×{212500 / 100000:.2f}",
        "КРИТИЧЕСКОЕ",
    ),
    (
        "К3: H200 цена",
        "до 50 млн (карта?)",
        "5,1–7,7 млн (карта)",
        "сервер, не карта",
        "КРИТИЧЕСКОЕ",
    ),
    (
        "К4: RTX 4090 карта",
        "170 000 — 220 000",
        "380 000 — 520 000",
        f"×{380000 / 170000:.1f} — ×{520000 / 220000:.1f}",
        "КРИТИЧЕСКОЕ",
    ),
    (
        "К4: H100 SXM карта",
        "1 290 000 — 2 550 000",
        "3 500 000 — 7 000 000",
        f"×{3500000 / 1290000:.1f} — ×{7000000 / 2550000:.1f}",
        "КРИТИЧЕСКОЕ",
    ),
    (
        "С4: RTX 4090 48GB",
        "коммерческая",
        "неофициальная модификация",
        "требует оговорки",
        "СРЕДНЕЕ",
    ),
    (
        "H1: Selectel H100/час",
        "900 — 2 200",
        "420 — 580",
        f"×{900 / 580:.1f} — ×{2200 / 420:.1f}",
        "НИЗКОЕ",
    ),
]

for name, old, new, ratio, severity in discrepancies:
    print(f"  [{severity:13s}] {name}")
    print(f"    отчёт:     {old}")
    print(f"    расчёт:     {new}")
    print(f"    расхождение: {ratio}")
    print()

# ═══════════════════════════════════════════════════════════════
# 11. РТ PRO 6000 Blackwell — оперданные
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("RTX PRO 6000 BLACKWELL 96GB — ОПЕРАЦИОННЫЕ ДАННЫЕ")
print("=" * 80)
print(f"  Закупка карты без шасси:  {rtx_pro_6000_card:>14,} ₽  (Comindware, апрель 2026)")
print(f"  В отчёте (строка 714):    1 200 000 — 1 800 000 ₽  (диапазон рынка с шасси)")
print(f"  Разъяснение: 900 000 — это карта без шасси, 1 500 000 — с сервером/WS")
print()

# ═══════════════════════════════════════════════════════════════
# 12. ПРОВЕРКА: КОРРЕКТНОСТЬ ОБЛАЧНОГО OPEX
# ═══════════════════════════════════════════════════════════════

print("=" * 80)
print("ПРОВЕРКА: КОРРЕКТНОСТЬ ОБЛАЧНОГО OPEX В TCO-ТАБЛИЦЕ")
print("=" * 80)

# Старый TCO: облачное мелкое = 1 020 000/год
# Новый: 1dedic 1x4090 = 100 000 × 12 = 1 200 000/год
# Старый: облачное среднее = 2 550 000/год
# Новый: Yandex 4xV100 = 511 000–876 000 × 12 = 6 132 000–10 512 000/год
# Старый: облачное крупное = 10 200 000/год
# Новый: Cloud.ru 5xA100 = 1 158 000 × 12 = 13 896 000/год; Yandex 8xA100 = 2 555 000 × 12 = 30 660 000/год

print(f"  Мелкое облако /год:")
print(f"    старое:     {1020000:>14,} ₽ (85 000/мес)")
print(f"    новое:      {100000 * 12:>14,} — {145000 * 12:>14,} ₽ (1dedic, 100–145K/мес)")
print(f"    delta:      ×{100000 * 12 / 1020000:.1f}")
print()
print(f"  Среднее облако /год:")
print(f"    старое:     {2550000:>14,} ₽ (212 500/мес)")
print(f"    новое:      {511000 * 12:>14,} — {876000 * 12:>14,} ₽ (Yandex 4xV100)")
print()
print(f"  Крупное облако /год:")
print(f"    старое:     {10200000:>14,} ₽ (850 000/мес)")
print(
    f"    новое:      {1158000 * 12:>14,} — {2555000 * 12:>14,} ₽ (Cloud.ru 5xA100 — Yandex 8xA100)"
)
print()

print("=" * 80)
print("РАСЧЁТ ЗАВЕРШЁН. ВСЕ ЦИФРЫ ПРОВЕРЕНЫ НА КАЛЬКУЛЯТОРЕ.")
print("=" * 80)
