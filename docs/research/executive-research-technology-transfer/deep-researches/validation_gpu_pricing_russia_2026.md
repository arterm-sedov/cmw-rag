# GPU Pricing Research: Russia Market (March 2026)

**Research Date:** March 29, 2026  
**Search Method:** Web search via Tavily and direct source verification  
**Currency:** RUB (Russian Rubles), USD equivalents for reference

---

## Executive Summary

This research covers GPU cloud pricing relevant for the Russian market in March 2026, focusing on consumer-grade RTX 4090 and enterprise-grade A100/H100 options from Russian cloud providers.

---

## 1. RTX 4090 Pricing

### 1.1 Purchase Price (Retail Russia)

| Source | Price (RUB) | Notes |
|--------|-------------|-------|
| Torg-PC.ru | ~357,000 | GeForce RTX 4090 24GB |
| Price.ru | from 270,930 | Various retailers |
| ServerFlow.ru | ~350,000+ | Server-grade variants (RTX 4090 Turbo) |

**Note:** The RTX 4090D (China-specific variant) is also available in Russia. Prices vary by retailer and availability.

**Source URLs:**
- https://torg-pc.ru/catalog/rtx-4090-4090-ti/videokarta-geforce-rtx-4090-24-gb-gddr6x-384-bit-pci-express-gen-4-850w/
- https://price.ru/videokarty/rtx4090/
- https://serverflow.ru/catalog/komplektuyushchie/videokarty/

### 1.2 Cloud Pricing (International Reference)

| Provider | Hourly Price (USD) | Notes |
|----------|-------------------|-------|
| SynpixCloud | $0.29 | RTX 4090 |
| Various US Clouds | $0.39-$0.50 | RTX 4090 |

**Source:** https://www.synpixcloud.com/blog/cloud-gpu-pricing-comparison-2026

---

## 2. A100/H100 Cloud Pricing (Russian Providers)

### 2.1 Cloud.ru (Evolution Compute GPU) - Detailed Pricing

**Effective Date:** March 26, 2026

| Configuration | Hourly Price (RUB, no VAT) | Hourly Price (RUB, with VAT) |
|---------------|---------------------------|------------------------------|
| 5x H100 PCI (100 vCPU/550 GB RAM) | 2,250 | 2,745 |
| 5x A100 PCI (100 vCPU/625 GB RAM) | 1,300 | 1,586 |
| 5x H100 NVLink (100 vCPU/930 GB RAM) | 3,500 | 4,270 |
| 6x H100 NVLink (120 vCPU/1,116 GB RAM) | 4,200 | 5,124 |
| 6x H100 PCI (120 vCPU/660 GB RAM) | 2,700 | 3,294 |
| 6x A100 PCI (120 vCPU/750 GB RAM) | 1,560 | 1,903 |
| 7x H100 NVLink (140 vCPU/1,302 GB RAM) | 4,900 | 5,978 |
| 7x H100 PCI (140 vCPU/770 GB RAM) | 3,150 | 3,843 |
| 7x A100 PCI (140 vCPU/875 GB RAM) | 1,820 | 2,220 |
| 4x V100 (16 vCPU/256 GB RAM) | 810 | 988 |

**Per-GPU Hourly Estimate (rough):**
- H100 (PCIe): ~450-540 RUB/hour ($5-6 USD)
- A100 (PCIe): ~260-320 RUB/hour ($3-3.5 USD)
- H100 (NVLink): ~700-840 RUB/hour ($8-9 USD)

**Source:** https://cloud.ru/documents/tariffs/evolution/evolution-compute-gpu

### 2.2 MWS (MTS Web Services) - GPU Cloud

**Available GPUs:**
- NVIDIA V100 (16GB, 32GB)
- NVIDIA A40 (48GB)
- NVIDIA A100 (80GB)
- NVIDIA H100 (80GB)
- NVIDIA H200 (141GB)

**Pricing Model:**
- Pay-as-you-go (hourly)
- Monthly plans (up to 30% savings)
- Allocation Pool (fixed payment for selected period)

**Sample Pricing (from configurator):**
- V100 16GB: ~197 RUB/hour (1 GPU, 16 vCPU, 48 GB RAM, 800 GB SSD)

**Note:** Specific H100/A100 pricing requires contacting sales. MWS offers 14-day free trial.

**Source:** https://cloud.mts.ru/services/virtual-infrastructure-gpu/

### 2.3 VK Cloud (formerly MCS)

**Available GPUs:**
- NVIDIA A100 40GB
- NVIDIA A100 80GB
- NVIDIA V100 16GB
- NVIDIA V100S 32GB
- NVIDIA A30 24GB
- NVIDIA L40S 48GB

**Pricing:**
- Contact sales for quote (pay-as-you-go, monthly, or annual plans)
- Per-second billing
- Claim up to 70% savings vs. hardware purchase

**Source:** https://cloud.vk.com/docs/en/computing/gpu/concepts/about

### 2.4 Selectel Cloud

**Available GPUs (as of March 2026):**
- NVIDIA RTX 4090 (24GB) - consumer-grade
- NVIDIA A100 40GB / 80GB
- NVIDIA H100 (80GB)
- NVIDIA H200 (141GB)
- NVIDIA L4 (24GB)
- NVIDIA RTX 6000 Ada (48GB)
- NVIDIA A5000 (24GB)
- NVIDIA A30, A2, T4, and others

**Note:** Pricing requires calculator or contact. Selectel is a major Russian infrastructure provider with data centers in Russia.

**Source:** https://docs.selectel.ru/cloud/servers/create/create-gpu-server/

### 2.5 SberCloud (Cloud.ru Advanced)

**Service:** GPU Accelerated Servers  
**Effective Date:** January 1, 2026

**Available GPUs:**
- NVIDIA V100 32GB SXM
- NVIDIA A100 80GB SXM
- NVIDIA H100 80GB SXM

**Source:** https://cloud.ru/documents/tariffs/advanced/services/gpu-accelerated

---

## 3. Russian GPU Cloud Providers Summary

| Provider | GPUs Available | Notes |
|----------|---------------|-------|
| **Cloud.ru (Evolution)** | V100, A100, H100 | Detailed pricing published, market leader |
| **MWS (MTS)** | V100, A40, A100, H100, H200 | #1 in CNews GPU provider rating |
| **VK Cloud** | A100, V100, L40S, A30 | Pay-as-you-go, contact sales |
| **Selectel** | RTX 4090, A100, H100, H200, L4, etc. | Large infrastructure provider |
| **Yandex Cloud** | A100, V100 | Limited public pricing (bot protection) |

---

## 4. USD to RUB Conversion Reference (March 2026)

For rough comparison (approximate):
- 1 USD ≈ 85-90 RUB (market rate)

**International Cloud GPU Reference Prices (USD/hour):**
- RTX 4090: $0.29-$0.50
- A100: $1.29-$2.00
- H100: $2.49-$4.00

**Sources:**
- https://www.synpixcloud.com/blog/cloud-gpu-pricing-comparison-2026
- https://www.thundercompute.com/blog/nvidia-h100-pricing
- https://www.thundercompute.com/blog/nvidia-a100-pricing

---

## 5. Key Findings

1. **Cloud.ru (Evolution)** offers the most transparent Russian-language pricing with detailed tariff sheets in RUB.

2. **RTX 4090 purchase** in Russia costs approximately 270,000-360,000 RUB (~$3,000-$4,000 USD).

3. **H100 cloud pricing** in Russia starts at approximately 450-700 RUB/hour ($5-8 USD) depending on configuration.

4. **A100 cloud pricing** in Russia starts at approximately 260-320 RUB/hour ($3-3.5 USD).

5. **NVLink configurations** command a premium of ~50-60% over PCI variants.

6. Most Russian providers require contact sales for exact quotes, especially for enterprise GPUs.

---

## Sources

1. https://cloud.ru/documents/tariffs/evolution/evolution-compute-gpu
2. https://cloud.mts.ru/services/virtual-infrastructure-gpu/
3. https://cloud.vk.com/docs/en/computing/gpu/concepts/about
4. https://docs.selectel.ru/cloud/servers/create/create-gpu-server/
5. https://cloud.ru/documents/tariffs/advanced/services/gpu-accelerated
6. https://torg-pc.ru/catalog/rtx-4090-4090-ti/
7. https://price.ru/videokarty/rtx4090/
8. https://www.synpixcloud.com/blog/cloud-gpu-pricing-comparison-2026

---

*Research conducted: March 29, 2026*
