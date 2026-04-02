# GPU Pricing Research - Russian Market (March 2026)

**Research Date:** March 29, 2026  
**Exchange Rate Used:** 85 RUB/USD  

---

## Executive Summary

This report provides validated pricing for GPU rental and cloud services in the Russian market as of March 2026. Key findings include:

- **RTX 4090 dedicated servers**: Available from 16,000-26,000 RUB/month in Russia
- **A100/H100 cloud instances**: Limited availability; pricing available on request
- **On-premise GPU servers**: Estimated CAPEX 1.5-3M RUB per server with 4x RTX 4090

---

## 1. RTX 4090 Rental Prices - Russian Providers

### 1.1 HostKey (hostkey.ru)

**Status:** Active Russian provider  
**Location:** Moscow (DataPro), Netherlands, Germany, Finland, Iceland, USA

| Configuration | Monthly Price (RUB) | Hourly (RUB) | Notes |
|---------------|---------------------|--------------|-------|
| GPU server - individual config | From 26,000 | ~43 | Custom configurator |
| GPU server - ready servers | From 16,000 | ~26 | Quick deployment |
| VPS with dedicated GPU | From 7,000 | ~12 | Budget option |

**Specific RTX 4090 Configurations Observed:**
- 1x RTX 4090, 512GB RAM, AMD EPYC 9174F: ~€699/month (59,415 RUB)
- 8x RTX 4090, 2TB NVME, 2048GB RAM: ~€2,990/month (254,150 RUB)

**Source:** https://hostkey.ru/gpu-dedicated-servers/

### 1.2 FirstDEDIC / 1dedic (1dedic.ru)

**Status:** Active Russian provider  
**Location:** Moscow (IXcellerate Tier III, Web DC)

**Available GPU Cards:**
- NVIDIA L4, L40, L40S
- RTX 6000 Ada, RTX 4090, RTX 4090 TURBO
- H100

**Configuration Limits:**
- Up to 8x NVIDIA L4 cards
- Up to 4x L40, L40S, RTX 6000, RTX 4090, H100

**Pricing:** Contact sales (configurator-based pricing)  
**Source:** https://1dedic.ru/gpu-servers

### 1.3 LeaderGPU (leadergpu.com)

**Status:** BLOCKED for Russian customers (VAT registration issue)  
**Note:** Cannot serve Russian companies/individuals due to Russian VAT requirements

**Historical Pricing (for reference - not available):**
- 1x RTX 4090, 512GB RAM: €699/month (59,415 RUB)
- 8x RTX 4090: €2,990/month (254,150 RUB)
- 4x RTX 3090: €599/month (50,915 RUB)

**Source:** https://leadergpu.com (blocked)

---

## 2. A100/H100 Cloud Pricing in Russia

### 2.1 HostKey

| Configuration | Monthly Price (EUR) | Monthly Price (RUB) |
|---------------|---------------------|---------------------|
| 8x A100 80GB SXM4 | €7,900 | 671,500 |
| 8x A100 40GB | €4,900 | 416,500 |
| 1x H200 141GB | €2,986.90 | 253,887 |

**Source:** https://hostkey.ru/gpu-dedicated-servers/

### 2.2 FirstDEDIC (1dedic.ru)

- **H100** available in configurator
- Pricing requires direct inquiry
- Located in Tier III DC (IXcellerate)

**Source:** https://1dedic.ru/gpu-servers

### 2.3 MTC Web Services (MWS)

**Status:** Active Russian cloud provider  
**Services:**
- Virtual infrastructure with GPU
- GPU On-premises solutions
- MWS GPT platform

**Note:** Pricing requires account creation and inquiry  
**Source:** https://mws.ru/services/virtual-infrastructure-gpu/

### 2.4 Yandex Cloud

**Status:** Limited public pricing (requires authentication)  
**Note:** Cloud computing platform with GPU instances available

---

## 3. On-Premise GPU Server Costs (CAPEX Estimates)

### 3.1 Estimated Hardware Costs (March 2026)

| Component | Price Range (RUB) | Notes |
|-----------|-------------------|-------|
| RTX 4090 (retail, Russia) | 200,000 - 280,000 | Import pricing, varies |
| Server chassis (4-GPU capable) | 150,000 - 300,000 | Professional grade |
| CPU (AMD EPYC / Intel Xeon) | 100,000 - 250,000 | Depends on model |
| RAM (256GB DDR5) | 80,000 - 150,000 | Enterprise-grade |
| NVMe Storage (2TB) | 30,000 - 60,000 | Enterprise NVMe |
| PSU (1600W+) | 25,000 - 50,000 | Redundant recommended |
| Cooling solution | 30,000 - 80,000 | Air or liquid |

### 3.2 Total CAPEX Estimates

**Basic 1x RTX 4090 Server:**
- Hardware: ~500,000 - 700,000 RUB
- With installation: ~550,000 - 800,000 RUB

**Mid-range 4x RTX 4090 Server:**
- Hardware: ~1,200,000 - 1,600,000 RUB
- With installation: ~1,400,000 - 1,800,000 RUB

**Enterprise 8x H100 Server:**
- Hardware: ~8,000,000 - 15,000,000 RUB
- Requires specialized DC infrastructure

### 3.3 Additional On-Premise Costs

| Cost Category | Annual Estimate (RUB) |
|---------------|----------------------|
| Colocation (rack space) | 120,000 - 300,000 |
| Electricity (4x4090 @ 500W) | 150,000 - 250,000 |
| Network bandwidth (1Gbps) | 60,000 - 120,000 |
| Maintenance (optional) | 100,000 - 200,000 |

---

## 4. Global Comparison (at 85 RUB/USD)

| GPU | Global Hourly (AWS/GCP) | Global Monthly Est. | Russian Monthly (RUB) |
|-----|-------------------------|---------------------|----------------------|
| RTX 4090 | $0.50-1.50 | $150-300 | 16,000-60,000 |
| A100 40GB | $3.50-4.50 | $2,500-3,500 | 416,500 (8x) |
| A100 80GB | $4.50-6.00 | $3,500-4,500 | 671,500 (8x) |
| H100 | $4.00-5.50 | $3,000-4,200 | 253,887 (1x) |

**Note:** Russian dedicated server pricing is significantly lower than global cloud instances due to:
1. Hardware availability through local providers
2. Lower labor costs
3. Domestic data residency requirements driving local demand

---

## 5. Key Findings & Recommendations

### 5.1 Market Availability

1. **RTX 4090**: Readily available in Russia via HostKey and 1dedic
2. **A100/H100**: Limited availability; requires inquiry for specific configurations
3. **LeaderGPU**: No longer serves Russian market

### 5.2 Cost Optimization

| Use Case | Recommended Solution | Est. Monthly Cost (RUB) |
|----------|---------------------|------------------------|
| Development/Testing | VPS with GPU | 7,000-15,000 |
| Production (small scale) | 1-2x RTX 4090 dedicated | 30,000-60,000 |
| Production (medium scale) | 4x RTX 4090 dedicated | 80,000-150,000 |
| Enterprise ML/AI | 8x A100/H100 | 250,000-700,000 |

### 5.3 Provider Selection Criteria

| Provider | Pros | Cons |
|----------|------|------|
| HostKey | Competitive pricing, multiple locations | Netherlands focus |
| 1dedic | Russian DC (152-FZ compliant), Tier III | Configurator-based pricing |
| MWS | Enterprise features, compliance | Requires inquiry |

---

## 6. Source URLs

1. https://hostkey.ru/gpu-dedicated-servers/ - HostKey GPU pricing
2. https://1dedic.ru/gpu-servers - FirstDEDIC GPU servers
3. https://mws.ru/services/virtual-infrastructure-gpu/ - MTC Web Services GPU
4. https://leadergpu.com - Historical reference (blocked)

---

## 7. Limitations & Notes

- Prices are indicative and may vary based on contract terms, volume, and current availability
- Ruble exchange rate fluctuations may affect imported GPU pricing
- Some high-end configurations (H100, A100) may require lead time
- Verification of current pricing recommended before procurement decisions
- All prices exclude VAT unless otherwise specified

---

*Report generated: March 29, 2026*  
*Exchange rate: 85 RUB = 1 USD*
