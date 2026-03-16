#!/usr/bin/env python3
"""
Evaluation script for anonymization pipeline.
Tests on mixed RU/EN IT support scenarios with ground truth annotations.

Total: 45 samples (20 Russian from jayguard + 25 English synthesized)
"""

import time
import re
from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# TEST SAMPLES WITH GROUND TRUTH
# =============================================================================

# Russian samples (adapted from jayguard-ner-benchmark + IT support context)
RU_SAMPLES = [
    {
        "text": "Здравствуйте, я Иван Иванов, сотрудник компании ООО ТехноСтрой. Мой номер телефона +7-900-123-45-67, email ivanov@tehnostroy.ru. Нужна помощь с доступом к серверу 192.168.1.100.",
        "entities": [
            {"text": "Иван Иванов", "type": "PERSON", "start": 14, "end": 25},
            {"text": "ООО ТехноСтрой", "type": "ORGANIZATION", "start": 40, "end": 53},
            {"text": "+7-900-123-45-67", "type": "PHONE", "start": 70, "end": 84},
            {"text": "ivanov@tehnostroy.ru", "type": "EMAIL", "start": 86, "end": 109},
            {"text": "192.168.1.100", "type": "IP_ADDRESS", "start": 144, "end": 157},
        ]
    },
    {
        "text": "Обращается Петров Алексей из компании Сбербанк. Телефон для связи: +7-901-234-56-78. Проблема с банковским приложением по адресу ул. Ленина 15, офис 305.",
        "entities": [
            {"text": "Петров Алексей", "type": "PERSON", "start": 13, "end": 25},
            {"text": "Сбербанк", "type": "ORGANIZATION", "start": 30, "end": 38},
            {"text": "+7-901-234-56-78", "type": "PHONE", "start": 62, "end": 76},
            {"text": "ул. Ленина 15", "type": "ADDRESS", "start": 116, "end": 128},
        ]
    },
    {
        "text": "Елена Козлова, директор магазина Магнит, звонила по поводу кассы. Номер магазина: г. Москва, ул. Пушкина 10. Тел: +7-902-333-44-55",
        "entities": [
            {"text": "Елена Козлова", "type": "PERSON", "start": 0, "end": 12},
            {"text": "Магнит", "type": "ORGANIZATION", "start": 29, "end": 35},
            {"text": "г. Москва, ул. Пушкина 10", "type": "ADDRESS", "start": 63, "end": 83},
            {"text": "+7-902-333-44-55", "type": "PHONE", "start": 89, "end": 103},
        ]
    },
    {
        "text": "Системный администратор Сидоров И.И. из отдела IT компании Яндекс. Рабочий телефон +7-903-444-55-66, внутренний 2345. Сервер находится по адресу datacenter2, IP 10.0.0.50",
        "entities": [
            {"text": "Сидоров И.И.", "type": "PERSON", "start": 23, "end": 34},
            {"text": "Яндекс", "type": "ORGANIZATION", "start": 48, "end": 53},
            {"text": "+7-903-444-55-66", "type": "PHONE", "start": 72, "end": 86},
            {"text": "10.0.0.50", "type": "IP_ADDRESS", "start": 137, "end": 146},
        ]
    },
    {
        "text": "Клиент Андрей Волков, email a.volkov@company.com, обратился с проблемой входа в систему. Нужна помощь, пароль истёк. Компания: Газпром.",
        "entities": [
            {"text": "Андрей Волков", "type": "PERSON", "start": 7, "end": 19},
            {"text": "a.volkov@company.com", "type": "EMAIL", "start": 21, "end": 41},
            {"text": "Газпром", "type": "ORGANIZATION", "start": 101, "end": 107},
        ]
    },
    {
        "text": "Менеджер Наталья Смирнова из офиса на Тверской улице дом 7. Её контактный email: n.smirnova@office.ru, телефон +7-904-555-66-77",
        "entities": [
            {"text": "Наталья Смирнова", "type": "PERSON", "start": 10, "end": 25},
            {"text": "Тверской улице дом 7", "type": "ADDRESS", "start": 41, "end": 59},
            {"text": "n.smirnova@office.ru", "type": "EMAIL", "start": 80, "end": 100},
            {"text": "+7-904-555-66-77", "type": "PHONE", "start": 102, "end": 116},
        ]
    },
    {
        "text": "Служба поддержки получила заявку от Иванова Сергея. Телефон: +7-905-111-22-33. Проблема с сервером БД по адресу 192.168.0.10, порт 5432.",
        "entities": [
            {"text": "Иванов Сергей", "type": "PERSON", "start": 33, "end": 45},
            {"text": "+7-905-111-22-33", "type": "PHONE", "start": 56, "end": 70},
            {"text": "192.168.0.10", "type": "IP_ADDRESS", "start": 105, "end": 117},
        ]
    },
    {
        "text": "Обращение от сотрудника: Дмитрий Кузнецов, ООО РЖД, телефон для связи +7-906-222-33-44. Нужен доступ к системе бронирования.",
        "entities": [
            {"text": "Дмитрий Кузнецов", "type": "PERSON", "start": 23, "end": 38},
            {"text": "ООО РЖД", "type": "ORGANIZATION", "start": 40, "end": 47},
            {"text": "+7-906-222-33-44", "type": "PHONE", "start": 68, "end": 82},
        ]
    },
    {
        "text": "Клиентка Анна Петрова из клиники Медси. Email: anna.petrova@medsi.ru, тел. +7-907-333-44-55. Офис: Москва, ул. Арбат 23.",
        "entities": [
            {"text": "Анна Петрова", "type": "PERSON", "start": 10, "end": 21},
            {"text": "Медси", "type": "ORGANIZATION", "start": 29, "end": 34},
            {"text": "anna.petrova@medsi.ru", "type": "EMAIL", "start": 45, "end": 65},
            {"text": "+7-907-333-44-55", "type": "PHONE", "start": 72, "end": 86},
            {"text": "Москва, ул. Арбат 23", "type": "ADDRESS", "start": 93, "end": 111},
        ]
    },
    {
        "text": "Техническая поддержка: Александра Морозова, email a.morozova@support.com, +7-908-444-55-66. Компания-клиент: СИБУР.",
        "entities": [
            {"text": "Александра Морозова", "type": "PERSON", "start": 21, "end": 36},
            {"text": "a.morozova@support.com", "type": "EMAIL", "start": 44, "end": 67},
            {"text": "+7-908-444-55-66", "type": "PHONE", "start": 69, "end": 83},
            {"text": "СИБУР", "type": "ORGANIZATION", "start": 99, "end": 104},
        ]
    },
    {
        "text": "Запрос от Марии Ивановой из ВТБ. Рабочий номер +7-909-555-66-77, email m.ivanova@vtb.ru. Проблема с терминалом оплаты.",
        "entities": [
            {"text": "Мария Иванова", "type": "PERSON", "start": 14, "end": 26},
            {"text": "ВТБ", "type": "ORGANIZATION", "start": 32, "end": 35},
            {"text": "+7-909-555-66-77", "type": "PHONE", "start": 47, "end": 61},
            {"text": "m.ivanova@vtb.ru", "type": "EMAIL", "start": 63, "end": 80},
        ]
    },
    {
        "text": "ИТ-специалист Игорь Соколов, компания Яндекса. Контакт: igor.sokolov@yandex-team.ru, телефон +7-910-666-77-88. IP сервера: 172.16.0.100",
        "entities": [
            {"text": "Игорь Соколов", "type": "PERSON", "start": 13, "end": 25},
            {"text": "Яндекса", "type": "ORGANIZATION", "start": 38, "end": 45},
            {"text": "igor.sokolov@yandex-team.ru", "type": "EMAIL", "start": 55, "end": 80},
            {"text": "+7-910-666-77-88", "type": "PHONE", "start": 93, "end": 107},
            {"text": "172.16.0.100", "type": "IP_ADDRESS", "start": 127, "end": 139},
        ]
    },
    {
        "text": "Звонок от Ольги Кузьминой из Сбера. Email: olga.kuzmina@sberbank.ru, тел +7-911-777-88-99. Офис на ул. Вавилова.",
        "entities": [
            {"text": "Ольга Кузьмина", "type": "PERSON", "start": 13, "end": 25},
            {"text": "Сбера", "type": "ORGANIZATION", "start": 31, "end": 36},
            {"text": "olga.kuzmina@sberbank.ru", "type": "EMAIL", "start": 44, "end": 67},
            {"text": "+7-911-777-88-99", "type": "PHONE", "start": 74, "end": 88},
            {"text": "ул. Вавилова", "type": "ADDRESS", "start": 98, "end": 110},
        ]
    },
    {
        "text": "Сотрудник техподдержки: Максим Петров, email m.petrov@helpdesk.ru, номер +7-912-888-99-00. Обслуживает клиента Тинькофф.",
        "entities": [
            {"text": "Максим Петров", "type": "PERSON", "start": 24, "end": 36},
            {"text": "m.petrov@helpdesk.ru", "type": "EMAIL", "start": 44, "end": 65},
            {"text": "+7-912-888-99-00", "type": "PHONE", "start": 67, "end": 81},
            {"text": "Тинькофф", "type": "ORGANIZATION", "start": 100, "end": 108},
        ]
    },
    {
        "text": "Заявка от Ирины Лебедевой, менеджера компании Магнит. Контактный телефон +7-913-999-00-11, email i.lebedeva@magnit.ru",
        "entities": [
            {"text": "Ирина Лебедева", "type": "PERSON", "start": 14, "end": 26},
            {"text": "Магнит", "type": "ORGANIZATION", "start": 41, "end": 47},
            {"text": "+7-913-999-00-11", "type": "PHONE", "start": 70, "end": 84},
            {"text": "i.lebedeva@magnit.ru", "type": "EMAIL", "start": 86, "end": 107},
        ]
    },
    {
        "text": "Обращение: Сергей Николаев, компания Газпромнефть. Телефон: +7-914-000-11-22. Сервер для мониторинга: 192.168.10.50",
        "entities": [
            {"text": "Сергей Николаев", "type": "PERSON", "start": 13, "end": 26},
            {"text": "Газпромнефть", "type": "ORGANIZATION", "start": 39, "end": 51},
            {"text": "+7-914-000-11-22", "type": "PHONE", "start": 72, "end": 86},
            {"text": "192.168.10.50", "type": "IP_ADDRESS", "start": 118, "end": 131},
        ]
    },
    {
        "text": "Клиент: Татьяна Соколова, email t.sokolova@company.net, телефон +7-915-111-22-33. Компания: Роснефть, офис СПб.",
        "entities": [
            {"text": "Татьяна Соколова", "type": "PERSON", "start": 7, "end": 21},
            {"text": "t.sokolova@company.net", "type": "EMAIL", "start": 29, "end": 51},
            {"text": "+7-915-111-22-33", "type": "PHONE", "start": 53, "end": 67},
            {"text": "Роснефть", "type": "ORGANIZATION", "start": 78, "end": 86},
            {"text": "СПб", "type": "ADDRESS", "start": 92, "end": 95},
        ]
    },
    {
        "text": "IT-инженер Алексей Орлов из компании Северсталь. Email: a.orlov@severstal.com, тел +7-916-222-33-44. Проблема с VPN.",
        "entities": [
            {"text": "Алексей Орлов", "type": "PERSON", "start": 13, "end": 25},
            {"text": "Северсталь", "type": "ORGANIZATION", "start": 38, "end": 47},
            {"text": "a.orlov@severstal.com", "type": "EMAIL", "start": 56, "end": 78},
            {"text": "+7-916-222-33-44", "type": "PHONE", "start": 84, "end": 98},
        ]
    },
    {
        "text": "Специалист Екатерина Зайцева, обслуживает клиента Сбертех. Email: e.zaitseva@sber-tech.ru, номер +7-917-333-44-55",
        "entities": [
            {"text": "Екатерина Зайцева", "type": "PERSON", "start": 13, "end": 28},
            {"text": "Сбертех", "type": "ORGANIZATION", "start": 47, "end": 54},
            {"text": "e.zaitseva@sber-tech.ru", "type": "EMAIL", "start": 63, "end": 85},
            {"text": "+7-917-333-44-55", "type": "PHONE", "start": 92, "end": 106},
        ]
    },
    {
        "text": "Звонок: Денис Белов из компании Лукойл. Тел: +7-918-444-55-66. Email: d.belov@lukoil.ru, IP адрес: 10.10.0.25",
        "entities": [
            {"text": "Денис Белов", "type": "PERSON", "start": 8, "end": 18},
            {"text": "Лукойл", "type": "ORGANIZATION", "start": 31, "end": 37},
            {"text": "+7-918-444-55-66", "type": "PHONE", "start": 53, "end": 67},
            {"text": "d.belov@lukoil.ru", "type": "EMAIL", "start": 74, "end": 91},
            {"text": "10.10.0.25", "type": "IP_ADDRESS", "start": 102, "end": 112},
        ]
    },
]

# English samples (synthesized IT support scenarios)
EN_SAMPLES = [
    {
        "text": "Hi, I'm John Smith from Acme Corp. My phone is 555-123-4567 and email is john.smith@acme.com. Need help with server 192.168.1.50.",
        "entities": [
            {"text": "John Smith", "type": "PERSON", "start": 7, "end": 17},
            {"text": "Acme Corp", "type": "ORGANIZATION", "start": 22, "end": 31},
            {"text": "555-123-4567", "type": "PHONE", "start": 45, "end": 56},
            {"text": "john.smith@acme.com", "type": "EMAIL", "start": 67, "end": 86},
            {"text": "192.168.1.50", "type": "IP_ADDRESS", "start": 110, "end": 122},
        ]
    },
    {
        "text": "Customer Sarah Johnson, email s.johnson@techfirm.com, phone +1-555-234-5678. Issue with banking app at address 123 Main St, Boston MA.",
        "entities": [
            {"text": "Sarah Johnson", "type": "PERSON", "start": 9, "end": 21},
            {"text": "s.johnson@techfirm.com", "type": "EMAIL", "start": 28, "end": 51},
            {"text": "+1-555-234-5678", "type": "PHONE", "start": 59, "end": 73},
            {"text": "123 Main St, Boston MA", "type": "ADDRESS", "start": 97, "end": 116},
        ]
    },
    {
        "text": "IT admin Michael Brown from Tech Solutions. Contact: m.brown@techsol.com, phone 555-345-6789. Server IP: 10.0.0.100.",
        "entities": [
            {"text": "Michael Brown", "type": "PERSON", "start": 10, "end": 22},
            {"text": "Tech Solutions", "type": "ORGANIZATION", "start": 30, "end": 43},
            {"text": "m.brown@techsol.com", "type": "EMAIL", "start": 54, "end": 75},
            {"text": "555-345-6789", "type": "PHONE", "start": 77, "end": 89},
            {"text": "10.0.0.100", "type": "IP_ADDRESS", "start": 109, "end": 119},
        ]
    },
    {
        "text": "Client David Wilson from GlobalTech. Email: d.wilson@globaltech.io, phone: +1-555-456-7890. Office: 456 Oak Ave, Seattle WA 98101.",
        "entities": [
            {"text": "David Wilson", "type": "PERSON", "start": 7, "end": 18},
            {"text": "GlobalTech", "type": "ORGANIZATION", "start": 27, "end": 36},
            {"text": "d.wilson@globaltech.io", "type": "EMAIL", "start": 44, "end": 67},
            {"text": "+1-555-456-7890", "type": "PHONE", "start": 75, "end": 89},
            {"text": "456 Oak Ave, Seattle WA 98101", "type": "ADDRESS", "start": 99, "end": 127},
        ]
    },
    {
        "text": "Support ticket from Emily Davis at Innovation Inc. Phone: 555-567-8901, email e.davis@innovation.io. Server: 172.16.0.50",
        "entities": [
            {"text": "Emily Davis", "type": "PERSON", "start": 22, "end": 32},
            {"text": "Innovation Inc", "type": "ORGANIZATION", "start": 36, "end": 49},
            {"text": "555-567-8901", "type": "PHONE", "start": 60, "end": 72},
            {"text": "e.davis@innovation.io", "type": "EMAIL", "start": 80, "end": 102},
            {"text": "172.16.0.50", "type": "IP_ADDRESS", "start": 120, "end": 131},
        ]
    },
    {
        "text": "Customer service request: Robert Taylor, company DataFlow Systems. Email r.taylor@dataflow.com, phone +1-555-678-9012.",
        "entities": [
            {"text": "Robert Taylor", "type": "PERSON", "start": 26, "end": 38},
            {"text": "DataFlow Systems", "type": "ORGANIZATION", "start": 48, "end": 63},
            {"text": "r.taylor@dataflow.com", "type": "EMAIL", "start": 65, "end": 88},
            {"text": "+1-555-678-9012", "type": "PHONE", "start": 90, "end": 104},
        ]
    },
    {
        "text": "IT specialist Jennifer Martinez from CloudTech. Contact: j.martinez@cloudtech.net, phone 555-789-0123. Server IP: 192.168.5.10",
        "entities": [
            {"text": "Jennifer Martinez", "type": "PERSON", "start": 14, "end": 29},
            {"text": "CloudTech", "type": "ORGANIZATION", "start": 38, "end": 47},
            {"text": "j.martinez@cloudtech.net", "type": "EMAIL", "start": 58, "end": 82},
            {"text": "555-789-0123", "type": "PHONE", "start": 84, "end": 96},
            {"text": "192.168.5.10", "type": "IP_ADDRESS", "start": 116, "end": 128},
        ]
    },
    {
        "text": "Client: Christopher Lee from Enterprise Solutions. Email c.lee@enterprise.com, phone: +1-555-890-1234. Location: 789 Pine Rd, Chicago IL.",
        "entities": [
            {"text": "Christopher Lee", "type": "PERSON", "start": 8, "end": 22},
            {"text": "Enterprise Solutions", "type": "ORGANIZATION", "start": 31, "end": 49},
            {"text": "c.lee@enterprise.com", "type": "EMAIL", "start": 52, "end": 74},
            {"text": "+1-555-890-1234", "type": "PHONE", "start": 82, "end": 96},
            {"text": "789 Pine Rd, Chicago IL", "type": "ADDRESS", "start": 107, "end": 127},
        ]
    },
    {
        "text": "Support request from Amanda White, tech company ByteWave. Phone: 555-901-2345, email a.white@bytewave.io. Server: 10.10.10.5",
        "entities": [
            {"text": "Amanda White", "type": "PERSON", "start": 23, "end": 34},
            {"text": "ByteWave", "type": "ORGANIZATION", "start": 48, "end": 55},
            {"text": "555-901-2345", "type": "PHONE", "start": 65, "end": 77},
            {"text": "a.white@bytewave.io", "type": "EMAIL", "start": 84, "end": 104},
            {"text": "10.10.10.5", "type": "IP_ADDRESS", "start": 122, "end": 132},
        ]
    },
    {
        "text": "Ticket from Daniel Harris, company StreamLine Corp. Contact: d.harris@streamline.co, +1-555-012-3456. Office at 321 Elm Street.",
        "entities": [
            {"text": "Daniel Harris", "type": "PERSON", "start": 16, "end": 28},
            {"text": "StreamLine Corp", "type": "ORGANIZATION", "start": 40, "end": 53},
            {"text": "d.harris@streamline.co", "type": "EMAIL", "start": 64, "end": 88},
            {"text": "+1-555-012-3456", "type": "PHONE", "start": 90, "end": 104},
            {"text": "321 Elm Street", "type": "ADDRESS", "start": 114, "end": 127},
        ]
    },
    {
        "text": "Customer: Lisa Anderson, email l.anderson@techcorp.com, phone 555-123-4567. Company: TechCorp Industries, address 555 Tech Blvd.",
        "entities": [
            {"text": "Lisa Anderson", "type": "PERSON", "start": 10, "end": 22},
            {"text": "l.anderson@techcorp.com", "type": "EMAIL", "start": 30, "end": 56},
            {"text": "555-123-4567", "type": "PHONE", "start": 64, "end": 76},
            {"text": "TechCorp Industries", "type": "ORGANIZATION", "start": 86, "end": 105},
            {"text": "555 Tech Blvd", "type": "ADDRESS", "start": 114, "end": 127},
        ]
    },
    {
        "text": "IT support: Mark Thompson from DataSync Inc. Email m.thompson@datasync.net, phone +1-555-234-5678. Server IP: 192.168.2.100",
        "entities": [
            {"text": "Mark Thompson", "type": "PERSON", "start": 13, "end": 25},
            {"text": "DataSync Inc", "type": "ORGANIZATION", "start": 34, "end": 45},
            {"text": "m.thompson@datasync.net", "type": "EMAIL", "start": 53, "end": 77},
            {"text": "+1-555-234-5678", "type": "PHONE", "start": 79, "end": 93},
            {"text": "192.168.2.100", "type": "IP_ADDRESS", "start": 112, "end": 125},
        ]
    },
    {
        "text": "Client request: Karen Moore, company QuantumTech. Email k.moore@quantumtech.com, phone 555-345-6789. Office: 100 Science Way.",
        "entities": [
            {"text": "Karen Moore", "type": "PERSON", "start": 15, "end": 24},
            {"text": "QuantumTech", "type": "ORGANIZATION", "start": 34, "end": 45},
            {"text": "k.moore@quantumtech.com", "type": "EMAIL", "start": 53, "end": 78},
            {"text": "555-345-6789", "type": "PHONE", "start": 80, "end": 92},
            {"text": "100 Science Way", "type": "ADDRESS", "start": 101, "end": 115},
        ]
    },
    {
        "text": "Support case: Steven Garcia from Apex Solutions. Phone: +1-555-456-7890, email s.garcia@apexsol.com. Server: 10.0.5.50",
        "entities": [
            {"text": "Steven Garcia", "type": "PERSON", "start": 14, "end": 25},
            {"text": "Apex Solutions", "type": "ORGANIZATION", "start": 34, "end": 46},
            {"text": "+1-555-456-7890", "type": "PHONE", "start": 56, "end": 70},
            {"text": "s.garcia@apexsol.com", "type": "EMAIL", "start": 78, "end": 100},
            {"text": "10.0.5.50", "type": "IP_ADDRESS", "start": 118, "end": 127},
        ]
    },
    {
        "text": "Customer: Betty Robinson from FlexiSystems. Email b.robinson@flexisys.net, phone 555-567-8901. Server IP: 172.16.10.25",
        "entities": [
            {"text": "Betty Robinson", "type": "PERSON", "start": 10, "end": 23},
            {"text": "FlexiSystems", "type": "ORGANIZATION", "start": 32, "end": 44},
            {"text": "b.robinson@flexisys.net", "type": "EMAIL", "start": 52, "end": 77},
            {"text": "555-567-8901", "type": "PHONE", "start": 79, "end": 91},
            {"text": "172.16.10.25", "type": "IP_ADDRESS", "start": 109, "end": 121},
        ]
    },
    {
        "text": "IT ticket: George Clark, company NextGen Tech. Phone +1-555-678-9012, email g.clark@nextgentech.io. Location: 250 Innovation Dr.",
        "entities": [
            {"text": "George Clark", "type": "PERSON", "start": 11, "end": 22},
            {"text": "NextGen Tech", "type": "ORGANIZATION", "start": 32, "end": 43},
            {"text": "+1-555-678-9012", "type": "PHONE", "start": 53, "end": 67},
            {"text": "g.clark@nextgentech.io", "type": "EMAIL", "start": 75, "end": 98},
            {"text": "250 Innovation Dr", "type": "ADDRESS", "start": 109, "end": 127},
        ]
    },
    {
        "text": "Client service: Nancy Lewis from CyberDyne Systems. Email n.lewis@cyberdyne.com, phone: 555-789-0123. Server at 192.168.20.10",
        "entities": [
            {"text": "Nancy Lewis", "type": "PERSON", "start": 16, "end": 26},
            {"text": "CyberDyne Systems", "type": "ORGANIZATION", "start": 35, "end": 51},
            {"text": "n.lewis@cyberdyne.com", "type": "EMAIL", "start": 59, "end": 82},
            {"text": "555-789-0123", "type": "PHONE", "start": 90, "end": 102},
            {"text": "192.168.20.10", "type": "IP_ADDRESS", "start": 120, "end": 133},
        ]
    },
    {
        "text": "Support request: Paul Walker, company VelocityStack. Contact: p.walker@velocitystack.net, +1-555-890-1234. Office: 75 Tech Park.",
        "entities": [
            {"text": "Paul Walker", "type": "PERSON", "start": 16, "end": 26},
            {"text": "VelocityStack", "type": "ORGANIZATION", "start": 40, "end": 52},
            {"text": "p.walker@velocitystack.net", "type": "EMAIL", "start": 62, "end": 88},
            {"text": "+1-555-890-1234", "type": "PHONE", "start": 90, "end": 104},
            {"text": "75 Tech Park", "type": "ADDRESS", "start": 114, "end": 125},
        ]
    },
    {
        "text": "Customer: Ruth Young, email r.young@innovatech.co, phone 555-901-2345. Company: InnovaTech LLC, address 800 Future Blvd.",
        "entities": [
            {"text": "Ruth Young", "type": "PERSON", "start": 10, "end": 20},
            {"text": "r.young@innovatech.co", "type": "EMAIL", "start": 28, "end": 51},
            {"text": "555-901-2345", "type": "PHONE", "start": 59, "end": 71},
            {"text": "InnovaTech LLC", "type": "ORGANIZATION", "start": 80, "end": 93},
            {"text": "800 Future Blvd", "type": "ADDRESS", "start": 102, "end": 116},
        ]
    },
    {
        "text": "IT case: Kevin King from CloudScale Inc. Email k.king@cloudscale.io, phone +1-555-012-3456. Server IP: 10.20.0.100",
        "entities": [
            {"text": "Kevin King", "type": "PERSON", "start": 11, "end": 20},
            {"text": "CloudScale Inc", "type": "ORGANIZATION", "start": 29, "end": 41},
            {"text": "k.king@cloudscale.io", "type": "EMAIL", "start": 49, "end": 71},
            {"text": "+1-555-012-3456", "type": "PHONE", "start": 73, "end": 87},
            {"text": "10.20.0.100", "type": "IP_ADDRESS", "start": 106, "end": 117},
        ]
    },
    {
        "text": "Client: Teresa Scott from NetFlow Systems. Email t.scott@netflow.net, phone 555-123-4567. Server: 192.168.100.5",
        "entities": [
            {"text": "Teresa Scott", "type": "PERSON", "start": 8, "end": 19},
            {"text": "NetFlow Systems", "type": "ORGANIZATION", "start": 28, "end": 41},
            {"text": "t.scott@netflow.net", "type": "EMAIL", "start": 49, "end": 69},
            {"text": "555-123-4567", "type": "PHONE", "start": 77, "end": 89},
            {"text": "192.168.100.5", "type": "IP_ADDRESS", "start": 107, "end": 120},
        ]
    },
    {
        "text": "Support: Brian Green, company HyperScale Tech. Phone: +1-555-234-5678, email b.green@hyperscale.com. Office 900 Cyber Lane.",
        "entities": [
            {"text": "Brian Green", "type": "PERSON", "start": 9, "end": 19},
            {"text": "HyperScale Tech", "type": "ORGANIZATION", "start": 29, "end": 42},
            {"text": "+1-555-234-5678", "type": "PHONE", "start": 52, "end": 66},
            {"text": "b.green@hyperscale.com", "type": "EMAIL", "start": 74, "end": 97},
            {"text": "900 Cyber Lane", "type": "ADDRESS", "start": 106, "end": 120},
        ]
    },
    {
        "text": "Ticket from Ashley Hill, company PrimeCloud. Email a.hill@primecloud.io, phone: 555-345-6789. Server IP: 172.16.50.10",
        "entities": [
            {"text": "Ashley Hill", "type": "PERSON", "start": 16, "end": 26},
            {"text": "PrimeCloud", "type": "ORGANIZATION", "start": 36, "end": 46},
            {"text": "a.hill@primecloud.io", "type": "EMAIL", "start": 54, "end": 76},
            {"text": "555-345-6789", "type": "PHONE", "start": 84, "end": 96},
            {"text": "172.16.50.10", "type": "IP_ADDRESS", "start": 114, "end": 126},
        ]
    },
    {
        "text": "Client service: Charles Adams from DevOps Pro. Contact: c.adams@devopspro.net, +1-555-456-7890. Office at 500 Code Street.",
        "entities": [
            {"text": "Charles Adams", "type": "PERSON", "start": 18, "end": 30},
            {"text": "DevOps Pro", "type": "ORGANIZATION", "start": 34, "end": 43},
            {"text": "c.adams@devopspro.net", "type": "EMAIL", "start": 54, "end": 77},
            {"text": "+1-555-456-7890", "type": "PHONE", "start": 79, "end": 93},
            {"text": "500 Code Street", "type": "ADDRESS", "start": 107, "end": 121},
        ]
    },
    {
        "text": "IT support: Laura Nelson from AgileStack. Email l.nelson@agilestack.com, phone 555-567-8901. Server: 10.100.0.25",
        "entities": [
            {"text": "Laura Nelson", "type": "PERSON", "start": 13, "end": 24},
            {"text": "AgileStack", "type": "ORGANIZATION", "start": 32, "end": 42},
            {"text": "l.nelson@agilestack.com", "type": "EMAIL", "start": 50, "end": 76},
            {"text": "555-567-8901", "type": "PHONE", "start": 84, "end": 96},
            {"text": "10.100.0.25", "type": "IP_ADDRESS", "start": 113, "end": 124},
        ]
    },
]

# Combine all samples
ALL_SAMPLES = RU_SAMPLES + EN_SAMPLES


# =============================================================================
# SIMPLE DETECTION FUNCTIONS (for evaluation)
# =============================================================================

def detect_regex(text: str) -> list[dict]:
    """Simple regex-based detection for evaluation."""
    entities = []
    
    # Phone patterns (RU + US)
    phone_patterns = [
        r'\+7\s*\(\d{3}\)\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}',
        r'\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}',
        r'\+7\d{10}',
        r'8\d{10}',
        r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
    ]
    
    for pattern in phone_patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "text": match.group(),
                "type": "PHONE",
                "start": match.start(),
                "end": match.end(),
                "source": "regex"
            })
    
    # Email
    for match in re.finditer(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text):
        entities.append({
            "text": match.group(),
            "type": "EMAIL",
            "start": match.start(),
            "end": match.end(),
            "source": "regex"
        })
    
    # IP Address
    for match in re.finditer(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text):
        entities.append({
            "text": match.group(),
            "type": "IP_ADDRESS",
            "start": match.start(),
            "end": match.end(),
            "source": "regex"
        })
    
    return entities


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_metrics(predicted: list[dict], ground_truth: list[dict], text: str) -> dict:
    """Calculate precision, recall, F1 for a single sample.
    
    Match using text overlap (not exact position).
    """
    gt_set = set()
    for gt in ground_truth:
        gt_set.add((gt["text"].lower(), gt["type"]))
    
    pred_set = set()
    for pred in predicted:
        pred_set.add((pred["text"].lower(), pred["type"]))
    
    # True positives: predicted that match ground truth
    true_positives = len(pred_set & gt_set)
    
    # False positives: predicted but not in ground truth
    false_positives = len(pred_set - gt_set)
    
    # False negatives: ground truth but not predicted
    false_negatives = len(gt_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation():
    """Run evaluation on all samples."""
    print("=" * 80)
    print("ANONYMIZATION PIPELINE EVALUATION")
    print("=" * 80)
    print(f"\nTotal samples: {len(ALL_SAMPLES)}")
    print(f"  - Russian: {len(RU_SAMPLES)}")
    print(f"  - English: {len(EN_SAMPLES)}")
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS (Regex-only baseline)")
    print("=" * 80)
    
    for i, sample in enumerate(ALL_SAMPLES):
        text = sample["text"]
        gt = sample["entities"]
        
        # Run detection
        predicted = detect_regex(text)
        
        # Calculate metrics
        metrics = calculate_metrics(predicted, gt, text)
        
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        
        # Show details for first few samples
        if i < 5 or metrics["f1"] < 0.5:
            lang = "RU" if i < len(RU_SAMPLES) else "EN"
            print(f"\n[{lang}] Sample {i+1}:")
            print(f"  Text: {text[:60]}...")
            print(f"  Predicted: {len(predicted)}, Ground Truth: {len(gt)}")
            print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
            print(f"  Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1']:.2f}")
    
    # Calculate overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("\n" + "=" * 80)
    print("OVERALL METRICS (Regex-only baseline)")
    print("=" * 80)
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"\nPrecision: {overall_precision:.4f}")
    print(f"Recall:    {overall_recall:.4f}")
    print(f"F1 Score:  {overall_f1:.4f}")
    
    print("\n" + "=" * 80)
    print("EXPECTED IMPROVEMENT WITH FULL PIPELINE")
    print("=" * 80)
    print("With 4-stage cascade (Regex + dslim + Gherman + EU-PII):")
    print("- Should detect PERSON entities (names)")
    print("- Should detect ORGANIZATION entities (companies)")
    print("- Should detect ADDRESS entities (locations)")
    print("- Expected F1 improvement: ~0.3-0.5 (depending on entity types)")
    
    return {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


if __name__ == "__main__":
    run_evaluation()
