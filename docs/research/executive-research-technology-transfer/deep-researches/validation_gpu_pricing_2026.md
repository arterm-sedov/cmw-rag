# GPU & AI Infrastructure Pricing Validation - March 2026

**Research Date:** March 29, 2026  
**Exchange Rate:** 85 RUB/USD (as specified)

---

## Executive Summary

This document validates current market pricing for GPU hardware and cloud AI infrastructure as of March 2026. Key findings indicate significant price differences between hyperscalers (AWS, GCP, Azure) and specialized GPU cloud providers, with the latter offering 3-6x cost savings. NVIDIA H100/H200 remain the gold standard for AI training, while AMD MI300X/MI355X provide competitive alternatives. Chinese alternatives like Huawei Atlas 350 have emerged as viable domestic options.

---

## 1. NVIDIA GPU Pricing

### 1.1 NVIDIA H100 (Hopper Architecture)

| Configuration | Purchase Price (USD) | Monthly Rental (USD) | RUB Equivalent |
|---------------|---------------------|---------------------|----------------|
| H100 PCIe 80GB | $25,000 - $30,000 | N/A | 2,125,000 - 2,550,000 |
| H100 SXM 80GB | $27,000 - $40,000 | N/A | 2,295,000 - 3,400,000 |
| 8-GPU DGX System | $200,000 - $216,000 | N/A | 17,000,000 - 18,360,000 |

**Cloud Rental (per GPU-hour):**
- Thunder Compute: $1.38/hr
- Vast.ai (marketplace): $1.53-2.27/hr
- RunPod: $1.99/hr
- Lambda Labs: $2.86/hr
- AWS p5.48xlarge (8xH100): $6.88-12.29/hr per GPU
- Azure ND H100 v5: $6.98-12.29/hr per GPU

**Sources:**
- https://www.thundercompute.com/blog/nvidia-h100-pricing (March 2026)
- https://docs.jarvislabs.ai/blog/h100-price (2026)
- https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/

### 1.2 NVIDIA H200 (Hopper Architecture)

| Configuration | Purchase Price (USD) | Monthly Rental (USD) | RUB Equivalent |
|---------------|---------------------|---------------------|----------------|
| H200 PCIe 141GB | $30,000 - $35,000 | N/A | 2,550,000 - 2,975,000 |
| H200 SXM 141GB | $38,000 - $45,000 | N/A | 3,230,000 - 3,825,000 |
| 8-GPU DGX H200 | $308,000 - $420,000 | N/A | 26,180,000 - 35,700,000 |

**Cloud Rental (per GPU-hour):**
- Jarvislabs: $3.80/hr
- Google Cloud A3-H200 (Spot): $3.72/hr
- Vast.ai: $2.43/hr
- AWS p5e.48xlarge (8xH200): $3.97-10.60/hr
- Azure Standard_ND96isr_H200_v5: $10.60/hr

**Sources:**
- https://www.szwecent.com/what-is-the-current-price-of-the-h200-gpu-in-2025/
- https://docs.jarvislabs.ai/blog/h200-price (2026)
- https://www.fluence.network/blog/nvidia-h200-deep-dive/

### 1.3 NVIDIA RTX 4090 (Consumer)

| Condition | Price (USD) | RUB Equivalent |
|-----------|-------------|---------------|
| New (Amazon) | $2,755 - $3,489 | 234,175 - 296,565 |
| Used (eBay) | $2,142 - $2,200 | 182,070 - 187,000 |
| MSRP (2022) | $1,599 | 135,915 |

**Notes:** RTX 4090 prices have increased 25-40% above MSRP due to US export restrictions and AI demand.

**Sources:**
- https://bestvaluegpu.com/history/new-and-used-rtx-4090-price-history-and-specs/
- https://www.tomshardware.com/pc-components/gpus/lowest-gpu-prices-tracking

---

## 2. AMD GPU Pricing

### 2.1 AMD Instinct MI300X

| Configuration | Purchase Price (USD) | Cloud Rental | RUB Equivalent |
|---------------|---------------------|--------------|---------------|
| MI300X 8-way Accelerator | $936,955 | N/A | 79,641,175 |
| Cloud (DigitalOcean) | N/A | $1.99/hr | 169.15/hr |
| Cloud (Hot Aisle) | N/A | $1.99/hr | 169.15/hr |
| Cloud (Vultr, spot) | N/A | $0.95/hr | 80.75/hr |
| Cloud (RunPod) | N/A | $0.50-3.00/hr | 42.50-255/hr |

**Specifications:** 192GB HBM3, 5,300 GB/s bandwidth, CDNA 3 architecture

**Sources:**
- https://getdeploying.com/gpus/amd-mi300x
- https://www.cdw.com/product/amd-instinct-mi300x-8-way-accelerator-graphics-card-instinct-mi300x-1/8238901

### 2.2 AMD Instinct MI355X

| Configuration | Purchase Price (USD) | Cloud Rental | RUB Equivalent |
|---------------|---------------------|--------------|---------------|
| MI355X 8-GPU (reserved 36mo) | N/A | $2.29/hr | 194.65/hr |
| MI355X 8-GPU (Oracle, on-demand) | N/A | $8.60/hr | 731/hr |
| MI355X 8-GPU (TensorWave) | N/A | On Request | On Request |

**Specifications:** 288GB HBM3E, 8TB/s bandwidth, CDNA 4 architecture, launched June 2025

**Sources:**
- https://getdeploying.com/gpus/amd-mi355x
- https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html

### 2.3 AMD Consumer GPUs (RX 7900 Series)

| Model | Current Price (USD) | Lowest-Ever (USD) | MSRP (USD) | RUB Equivalent |
|-------|--------------------|--------------------|-------------|----------------|
| RX 7900 XTX | $1,169 | $749 | $999 | 99,365 |
| RX 7900 XT | $669-1,129 | $559 | $899 | 56,865-95,965 |
| Used RX 7900 XTX (eBay) | ~$575-839 | N/A | N/A | 48,875-71,315 |

**Sources:**
- https://www.tomshardware.com/pc-components/gpus/lowest-gpu-prices-tracking
- https://bestvaluegpu.com/history/new-and-used-rx-7900-xt-price-history-and-specs/

---

## 3. Chinese GPU Alternatives

### 3.1 Huawei Atlas 350

| Specification | Value |
|---------------|-------|
| Chip | Ascend 950PR |
| Memory | 112GB HBM (HiBL 1.0) |
| FP4 Performance | 1.56 PFLOPS |
| Memory Bandwidth | 1.4 TB/s |
| TDP | 600W |
| **Estimated Price** | **~$16,000 (111,000 yuan)** |
| RUB Equivalent | 1,360,000 |

**Comparison:**
- vs NVIDIA H20: 2.87x performance, ~50% of price
- vs NVIDIA H200: ~80% performance, ~40% of price

**Notes:** First Chinese AI accelerator with FP4 support. Positioned as domestic alternative free from US export controls. Announced March 20, 2026 at Huawei China Partner Conference.

**Sources:**
- https://www.chosun.com/english/industry-en/2026/03/24/VLDBXXKYVJDLDHXIC43PPLN4IQ/
- https://www.tomshardware.com/pc-components/gpus/huawei-unveils-new-atlas-350-ai-accelerator-with-1-56-pflops-of-fp4-compute-and-up-to-112gb-of-hbm-claims-2-8x-more-performance-than-nvidias-h20
- https://hyper.ai/en/stories/56b812d9c9cbec3856c100229cfc6914

---

## 4. Cloud GPU Pricing - Hyperscalers

### 4.1 AWS (Amazon Web Services)

| Instance | GPUs | On-Demand/hr (USD) | Spot/hr (USD) | RUB Equivalent |
|----------|------|-------------------|---------------|----------------|
| p5.48xlarge | 8x H100 | $98.32 | ~$44-55 | 8,357-8,357 |
| p5e.48xlarge | 8x H200 | $39.80-84.80 | N/A | 3,383-7,208 |
| g6.xlarge | 1x A10G | $0.80 | $0.39 | 68.30 |
| g6.16xlarge | 8x L4 | $6.44 | N/A | 547.40 |

**Sources:**
- https://wring.co/blog/aws-gpu-instance-pricing-guide
- https://aws.amazon.com/ec2/instance-types/p5/
- https://instances.vantage.sh/aws/ec2/g6.xlarge

### 4.2 Google Cloud Platform (GCP)

| Instance | GPUs | On-Demand/hr (USD) | Spot/hr (USD) | 3-Year CUD (USD) |
|----------|------|-------------------|---------------|-------------------|
| a3-highgpu-1g | 1x H100 | $10.98 | $3.69 | N/A |
| a3-highgpu-8g | 8x H100 | $87.84 | ~$29.52 | N/A |
| a3-ultragpu-8g | 8x H200 | $86.76-133.20 | N/A | $3,546-4,088/mo |
| g2-standard-4 | 1x L4 | $0.70 | N/A | N/A |

**Sources:**
- https://cloudprice.net/gcp/compute/instances/a3-highgpu-1g
- https://cloud.google.com/compute/gpus-pricing
- https://gcloud-compute.com/a3-highgpu-1g.html

### 4.3 Microsoft Azure

| Instance | GPUs | On-Demand/hr (USD) | RUB Equivalent |
|----------|------|-------------------|----------------|
| ND H100 v5 | 1x H100 | $6.98-12.29 | 593-1,045 |
| ND96isr_H200_v5 | 8x H200 | $84.80 | 7,208 |
| NVadsA10v5 | 1x A10 | $0.45-1.60 | 38-136 |

**Sources:**
- https://verda.com/blog/cloud-gpu-pricing-comparison
- https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/
- https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvadsa10v5-series

---

## 5. Specialized GPU Cloud Providers (Cost Comparison)

| Provider | H100/hr (USD) | H200/hr (USD) | A100 80GB/hr | Notes |
|----------|---------------|---------------|--------------|-------|
| Thunder Compute | $1.38 | N/A | $0.78 | Lowest on-market |
| Vast.ai | $1.53-2.27 | $2.43 | $0.90-1.20 | Marketplace |
| RunPod | $1.99-2.69 | N/A | $1.50 | Community Cloud |
| Lambda Labs | $2.86-3.44 | $3.79 | $1.29 | On-demand |
| CoreWeave | $6.16 | N/A | N/A | HGX H100 |
| Spheron | $2.01 | $4.54 | $1.07 | Multi-cloud |
| Paperspace | $5.95 | N/A | $3.18 | Enterprise |

**Key Insight:** Hyperscalers (AWS, GCP, Azure) charge 3-6x more than specialized providers for equivalent hardware.

**Sources:**
- https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/
- https://www.thundercompute.com/blog/nvidia-h100-pricing (March 2026)

---

## 6. Russian Cloud GPU Providers

### 6.1 Yandex Cloud

**Available GPUs:** NVIDIA T4, V100, A100  
**Pricing Model:** Per-second billing with sustained-use discounts  
**Data Residency:** Russia region compliant  
**Features:** Distributed training infrastructure, managed ML services

**Notes:** Yandex Cloud offers GPU instances but specific pricing requires direct inquiry. Positioned for enterprises requiring data residency in Russia.

**Sources:**
- https://dataoorts.com/top-5-plus-gpu-cloud-providers-in-russia/
- https://yandex.cloud/en/docs/compute/pricing

### 6.2 Reg.Ru Cloud

| Configuration | Price (RUB/hr) | Price (USD/hr) |
|---------------|----------------|----------------|
| NVIDIA 1xA100 80GB | 419.74 | ~$4.94 |
| (16vCPU/128GB/512GB) | 282,064/mo | ~$3,319/mo |

**Sources:**
- https://www.whtop.com/compare/cloud.yandex.com,reg.ru

### 6.3 RUVDS

**Services:** VPS/VDS hosting, no dedicated GPU instances found  
**Starting Prices:** From 139 RUB/month (non-GPU VPS)  
**Data Centers:** 20 global locations including Moscow, London, Frankfurt

**Sources:**
- https://ruvds.com/en-usd

### 6.4 1Dedicated

**Services:** Dedicated servers, limited GPU offerings  
**Target:** Enterprise hosting, not specialized AI workloads

**Sources:**
- https://www.whtop.com/compare/1dedic.ru,ruvds

---

## 7. Summary Table: GPU Pricing Comparison

### Purchase Prices (USD)

| GPU | Price | RUB | Primary Use |
|-----|-------|-----|-------------|
| NVIDIA H100 80GB | $25,000-40,000 | 2.1-3.4M | Enterprise AI Training |
| NVIDIA H200 141GB | $30,000-45,000 | 2.55-3.83M | Large Model Inference |
| NVIDIA RTX 4090 | $2,755-3,489 | 234K-297K | Consumer/Edge AI |
| AMD MI300X | ~$10,000-15,000* | 850K-1.28M | Enterprise AI |
| AMD MI355X | TBA | TBA | High-Performance AI |
| Huawei Atlas 350 | ~$16,000 | 1.36M | China Market AI |
| AMD RX 7900 XTX | $1,169 | 99K | Consumer Gaming |

*Estimated from 8-way pricing

### Cloud Rental (USD/GPU-hour)

| GPU | Specialized Cloud | Hyperscalers | Savings |
|-----|-------------------|--------------|---------|
| H100 | $1.38-3.00 | $6.88-12.29 | 3-6x |
| H200 | $2.43-4.54 | $10.60-14.24 | 2-4x |
| A100 80GB | $0.78-1.50 | $2.00-4.00 | 2-3x |
| RTX 4090 | $0.30-0.50 | N/A | N/A |

---

## 8. Key Findings

1. **NVIDIA Dominance:** H100/H200 remain the gold standard but at premium pricing
2. **Hyperscaler Premium:** AWS/GCP/Azure charge 3-6x more than specialized GPU clouds
3. **AMD Viability:** MI300X/MI355X competitive for memory-intensive workloads
4. **Chinese Alternative:** Huawei Atlas 350 offers ~80% H200 performance at ~40% cost
5. **Edge Inference:** Consumer RTX 4090 viable for smaller workloads at $0.30-0.50/hr cloud
6. **Russian Market:** Limited dedicated GPU cloud options; reg.ru offers A100 at ~$5/hr

---

## 9. Sources & References

1. https://www.thundercompute.com/blog/nvidia-h100-pricing - March 2026
2. https://docs.jarvislabs.ai/blog/h100-price - 2026
3. https://docs.jarvislabs.ai/blog/h200-price - 2026
4. https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/
5. https://getdeploying.com/gpus/amd-mi300x
6. https://getdeploying.com/gpus/amd-mi355x
7. https://www.tomshardware.com/pc-components/gpus/huawei-unveils-new-atlas-350-ai-accelerator-with-1-56-pflops-of-fp4-compute-and-up-to-112gb-of-hbm-claims-2-8x-more-performance-than-nvidias-h20
8. https://www.chosun.com/english/industry-en/2026/03/24/VLDBXXKYVJDLDHXIC43PPLN4IQ/
9. https://bestvaluegpu.com/history/new-and-used-rtx-4090-price-history-and-specs/
10. https://www.tomshardware.com/pc-components/gpus/lowest-gpu-prices-tracking
11. https://aws.amazon.com/ec2/instance-types/p5/
12. https://cloud.google.com/compute/gpus-pricing
13. https://verda.com/blog/cloud-gpu-pricing-comparison
14. https://dataoorts.com/top-5-plus-gpu-cloud-providers-in-russia/
15. https://www.whtop.com/compare/cloud.yandex.com,reg.ru
