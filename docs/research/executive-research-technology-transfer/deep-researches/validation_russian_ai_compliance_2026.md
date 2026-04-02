# Russian AI and Data Protection Compliance Requirements (March 2026)

**Research Date:** March 2026  
**Classification:** Executive Research - Technology Transfer  
**Sources:** Russian and English regulatory sources, legal analyses, government publications

---

## Executive Summary

This report provides a comprehensive analysis of Russian data protection and AI compliance requirements as of March 2026. Key developments include significant amendments to Federal Law No. 152-FZ "On Personal Data" through Federal Law No. 23-FZ (effective July 2025), ongoing development of dedicated AI legislation, and increased enforcement activity by Roskomnadzor. Organizations deploying AI systems that process personal data of Russian citizens must navigate a complex regulatory landscape combining existing data protection law with emerging AI-specific requirements.

---

## 1. Current Legal Framework Status

### 1.1 Primary Legislation: Federal Law No. 152-FZ

The foundation of Russian data protection law remains **Federal Law No. 152-FZ "On Personal Data"** (dated July 27, 2006). This law governs all processing of personal data within Russia and extraterritorially for Russian citizens.

**Key amendments in force as of 2026:**

| Amendment | Effective Date | Key Changes |
|-----------|---------------|-------------|
| Federal Law No. 23-FZ | July 1, 2025 | Stricter localization requirements |
| Federal Law No. 23-FZ | September 1, 2025 | Anonymization requirements for data operators |
| Federal Law No. 233-FZ | August 8, 2024 | Experimental legal regime for AI |
| Federal Law No. 266-FZ | March 1, 2023 | Enhanced data subject rights |

### 1.2 AI-Specific Legislation Status

As of March 2026, **Russia does not have a comprehensive dedicated AI law**. The regulatory approach combines:

1. **152-FZ amendments** - Personal data aspects of AI processing
2. **Experimental Legal Regime (ELR)** - Federal Law No. 233-FZ (August 2024) establishing a regulatory sandbox for AI companies
3. **Pending Framework Bill** - The Ministry of Digital Development has published proposals for a comprehensive AI law "On the Basics of State Regulation of the Application of Artificial Intelligence Technologies" - currently in consultation phase (as of March 2026)

The draft AI law would impose conditions on developers and operators of AI models including:
- Ensuring AI behavior does not lead to discrimination
- Prohibition on use for prohibited purposes
- Requirements to stop work in the event of threats to human life/rights

**Government Structure for AI (2026):**
- Presidential AI development stimulation headquarters created
- FSTEC and FSB to issue certificates for AI systems at critical facilities
- Interdepartmental group created to combat deepfakes

---

## 2. Specific Requirements for AI Systems Handling Personal Data

### 2.1 Legal Basis for Processing

Under Article 6 of 152-FZ, AI systems processing personal data require one of the following legal bases:

1. **Consent** - Explicit consent from the data subject
2. **Contract** - Performance of a contract with the data subject
3. **Legal obligation** - Compliance with Russian law
4. **Vital interests** - Protection of life/health of the data subject
5. **Public interest** - Exercise of government powers

For AI/ML systems, consent and contract are the most common bases.

### 2.2 Automated Decision-Making Rights

Data subjects have specific rights under 152-FZ regarding automated processing:

- **Right to object** to profiling and automated decision-making
- **Right to request human intervention** in automated decisions affecting their rights
- **Right to contest decisions** made solely by automated processing

Organizations must provide meaningful information about the logic involved in automated decision-making.

### 2.3 Anonymization Requirements (Effective September 1, 2025)

Federal Law No. 23-FZ introduced stringent anonymization requirements:

- **Irreversibility requirement**: Anonymization must be irreversible
- **Separation requirement**: Anonymized data must not be stored alongside original data
- **Method confidentiality**: Disclosure of anonymization methods to third parties is prohibited
- **Roskomnadzor guidance**: Updated requirements for pseudonymization and data alteration techniques

**Practical implication for AI/ML**: Training data used for AI model development must be properly anonymized before use, and anonymization methods must be kept confidential.

### 2.4 Cross-Border Transfer Restrictions

**Current requirements (post-July 2025):**

- Personal data of Russian citizens must first be collected and stored in databases located in Russia
- Cross-border transfers are permitted only after initial processing through Russian databases
- Notification to Roskomnadzor is required before transfer
- Recipient country must provide "adequate level of protection"
- Transfers to "unfriendly countries" face additional restrictions

---

## 3. Data Localization Requirements Relevant to AI/ML

### 3.1 Core Localization Requirements (Article 18 of 152-FZ)

**Effective July 1, 2025** - Part 5 of Article 18 establishes new wording:

When personal data is collected (including via the Internet), operators **must ensure** the following operations use databases located in Russia:
- Recording
- Systematization
- Capture
- Storage
- Refinement (updating, amendment)
- Retrieval

**Prohibition**: These operations using databases outside Russia are explicitly prohibited for Russian citizens' personal data.

### 3.2 Requirements for AI Model Training

| Requirement | Description |
|-------------|-------------|
| Training data storage | Training datasets containing Russian personal data must be stored in Russia |
| Processing location | Initial processing of personal data for AI training must occur in Russia |
| Cross-border transfer | Only anonymized data may be transferred abroad after initial Russian processing |
| Anonymization | Must meet irreversible standard before any transfer |

### 3.3 Government Data Sharing (Effective September 1, 2025)

New requirement: All personal data operators must provide anonymized datasets to designated government information systems upon request from Roskomnadzor.

This has implications for:
- AI companies with training datasets
- Organizations with large personal data holdings
- Any entity developing AI models using Russian personal data

---

## 4. Industry-Specific Requirements

### 4.1 Financial Sector

**Key requirements:**
- Enhanced security requirements for banking/financial data
- Central Bank of Russia regulations on data handling for fintech
- Mandatory breach notification to Central Bank
- Special requirements for biometric data in financial services
- Localization requirements apply with no exceptions for financial data

**AI-specific considerations:**
- Automated credit scoring systems require explicit consent
- Algorithmic trading systems subject to additional disclosure requirements
- Customer profiling for marketing requires opt-out mechanisms

### 4.2 Healthcare

**Key requirements:**
- Health data classified as "special categories" under 152-FZ
- Stricter consent requirements for medical data processing
- Requirements for medical AI diagnostic systems
- Integration with state healthcare information systems

**AI-specific considerations:**
- Clinical decision support systems must have human oversight
- Medical AI must comply with Roszdravnadzor requirements
- Patient consent required for AI-assisted diagnostics

### 4.3 Additional Sector Notes

- **Telecom**: Additional Roskomnadzor oversight for customer data
- **E-commerce**: Localization requirements for customer transaction data
- **Education**: Proposed restrictions on AI use in education (2025-2026)

---

## 5. Roskomnadzor Requirements and Guidance

### 5.1 Regulatory Authority

Roskomnadzor (Federal Service for Supervision of Communications, Information Technology, and Mass Media) is the primary data protection regulator in Russia.

**Key responsibilities:**
- Registration of personal data operators
- Cross-border transfer approvals
- Enforcement of 152-FZ requirements
- Anonymization technique standards

### 5.2 Current Guidance (2025-2026)

Roskomnadzor has issued guidance on:
- Anonymization techniques (pseudonymization, data alteration)
- Consent requirements for AI processing
- Data operator notification procedures
- Cross-border transfer mechanisms

### 5.3 AI Internet Filtering (2026)

Roskomnadzor is implementing AI-powered internet traffic filtering system planned for 2026, which will strengthen content censorship capabilities. Organizations should be aware of:
- Increased monitoring of online data processing
- Automated detection of compliance violations
- Potential expansion of blocked content categories

### 5.4 Proposed Foreign AI Tool Restrictions (2026)

Ministry for Digital Development proposals (March 2026):
- Foreign AI applications ("cross-border AI tools") would require compliance with Russian regulations
- Potential ban or restriction for non-compliant foreign AI tools
- Includes popular tools like Claude, ChatGPT, Gemini
- Undergoing government approval process

---

## 6. Enforcement Actions and Penalties

### 6.1 Enforcement Mechanisms

Roskomnadzor can:
- Issue enforcement notices requiring correction of violations
- Impose administrative fines
- Block websites and online content
- Refer cases for civil or criminal liability

### 6.2 Recent Enforcement Activity

Based on available sources:
- Increased enforcement actions against companies for data breach notification failures
- Fines for lack of proper data security measures
- Website blocking for personal data processing violations
- Active monitoring of cross-border data transfers

### 6.3 Penalty Structure

| Violation Type | Potential Penalty |
|---------------|-------------------|
| Failure to notify Roskomnadzor | Administrative fine |
| Unauthorized cross-border transfer | Up to 18 million RUB (legal entities) |
| Failure to implement data security measures | Variable, up to 18 million RUB |
| Processing without consent | Variable, case-by-case |

---

## 7. Practical Compliance Recommendations

### 7.1 For AI System Developers

1. **Data handling**: Ensure all personal data used in AI training is processed through Russian databases first
2. **Anonymization**: Implement irreversible anonymization meeting Roskomnadzor standards before any cross-border transfer
3. **Consent mechanisms**: Establish clear consent flows for personal data used in AI systems
4. **Transparency**: Provide clear information about automated decision-making logic
5. **Human oversight**: Maintain human review mechanisms for AI decisions affecting individuals

### 7.2 For Organizations Deploying AI

1. **Legal basis**: Document the legal basis for all personal data processing in AI systems
2. **Data mapping**: Map all personal data flows, especially cross-border transfers
3. **Localization**: Ensure Russian citizen data is stored in Russia before any processing
4. **Rights management**: Implement processes to handle data subject requests regarding automated decisions
5. **Vendor assessment**: Evaluate AI vendors for compliance with 152-FZ requirements

### 7.3 Monitoring Requirements

- Track proposed AI legislation developments
- Monitor Roskomnadzor guidance updates
- Review foreign AI tool restrictions when finalized
- Stay informed about enforcement trends

---

## 8. Key Regulatory Developments Timeline

| Date | Development |
|------|-------------|
| August 2024 | Federal Law No. 233-FZ - AI experimental legal regime |
| February 2025 | Federal Law No. 23-FZ signed |
| July 1, 2025 | Stricter localization requirements effective |
| September 1, 2025 | Anonymization requirements effective |
| 2026 (Q1) | AI framework bill in consultation |
| 2026 | Proposed foreign AI tool restrictions |
| 2026 | AI-powered internet filtering planned |

---

## 9. Sources and References

### Primary Russian Legal Sources
- Federal Law No. 152-FZ "On Personal Data" (July 27, 2006)
- Federal Law No. 23-FZ (February 28, 2025) - Amendments to 152-FZ
- Federal Law No. 233-FZ (August 8, 2024) - AI experimental regime

### Secondary Sources
- Roskomnadzor official guidance documents
- Ministry of Digital Development publications
- Russian legal practitioner analyses (Acsour, ALRUD, Lidings, etc.)

### International Sources
- Digital Policy Alert tracking
- Industry legal analyses (White & Case, Debevoise & Plimpton)
- Comparative data protection resources

---

## 10. Conclusion

The Russian regulatory landscape for AI and personal data is evolving rapidly. Organizations must navigate:

1. **Existing 152-FZ framework** with significant 2025 amendments
2. **Emerging AI-specific requirements** through the experimental legal regime
3. **Pending comprehensive AI legislation** expected in coming years
4. **Strict localization requirements** with no exceptions for AI training data
5. **Increased enforcement** by Roskomnadzor

**Key takeaway**: As of March 2026, there is no single "Russian AI law." Instead, AI systems handling personal data must comply with 152-FZ requirements (as amended), with additional requirements coming through the experimental legal regime and pending legislation. The localization-first approach requires careful architectural planning for any AI system processing Russian personal data.

---

*This report is based on publicly available sources as of March 2026. Organizations should verify current requirements with qualified Russian legal counsel before making compliance decisions.*
