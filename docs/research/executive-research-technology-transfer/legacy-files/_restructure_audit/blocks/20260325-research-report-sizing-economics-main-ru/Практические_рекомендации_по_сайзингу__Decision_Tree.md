## Практические рекомендации по сайзингу (Decision Tree)

### Decision Tree для выбора инфраструктуры

```
START
  │
  ├─ Утилизация < 40%?
  │   └─ ДА → Облако (OpEx model)
  │
  ├─ Утилизация > 60%?
  │   └─ ДА → On-Prem (CapEx model)
  │
  ├─ Данные должны оставаться локально?
  │   └─ ДА → Sovereign AI (On-Prem или российское облако)
  │
  ├─ Бюджет < 850 000 руб.?
  │   └─ ДА → RTX 4090 workstation (ориентир **< 10 000 USD × 85 ₽/USD**)
  │
  ├─ Бюджет 850 000 – 4 250 000 руб.?
  │   └─ ДА → A100/RTX 6000 workstation (**10–50 тыс. USD × 85**)
  │
  └─ Бюджет > 4 250 000 руб.?
      └─ ДА → Multi-GPU server или cloud cluster (**> 50 тыс. USD × 85**)
```

### Калькулятор TCO (упрощённый)

**Формула TCO (On-Prem):**
```
TCO = CapEx + (OpEx × Years) + (Energy × PUE × Years × Hours) + Personnel
```

**Формула TCO (Cloud):**
```
TCO = Hourly_Rate × 24 × 365 × Years + Egress_Fees + Storage_Fees
```

**Break-even (On-Prem vs Cloud):**
```
Break_even_months = CapEx / (Cloud_Monthly - On_Prem_Monthly_OpEx)
```

---
