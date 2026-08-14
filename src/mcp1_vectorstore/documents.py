"""Knowledge base: generic Telecom Revenue Assurance (RA) concepts.

Content is written as original, generic summaries informed by widely-used
industry framing — notably TM Forum's GB941 Revenue Assurance guidebook and
maturity model, the industry standards body that formalized RA practice.
No proprietary, company-specific, or verbatim licensed text is included.
"""

DOCUMENTS = [
    "Revenue Assurance (RA) is the telecom industry discipline focused on "
    "detecting, investigating, correcting, and preventing revenue and cost "
    "leakage across the network-to-billing chain. It emerged in the early "
    "1990s when carriers noticed that usage systems and billing systems "
    "often disagreed, and matured into a formal practice standardized by "
    "industry bodies such as TM Forum through frameworks like GB941.",
    "Revenue leakage is revenue that is earned but never billed or "
    "collected. Common causes include unrated call detail records, "
    "incorrect tariff plan configuration, system outages during mediation, "
    "configuration mismatches between network and billing systems, and "
    "manual adjustment errors. Leakage differs from fraud in that it is "
    "usually unintentional — a process or system failure rather than "
    "deliberate abuse.",
    "A Call Detail Record (CDR) is a data record produced by a telecom "
    "switch or network element that captures the details of a call, SMS, "
    "or data session — timestamp, duration, origin, destination, and cell "
    "site. CDRs move through mediation, rating, and billing stages before "
    "becoming a billed charge; a CDR lost or malformed at any stage is a "
    "common source of leakage.",
    "Billing vs. usage reconciliation compares the volume and value of "
    "usage recorded at the network or switch level against what was "
    "actually billed to the customer. Mismatches between these two "
    "datasets are a primary signal of revenue leakage and are typically "
    "reconciled on a daily or monthly cadence as part of a revenue "
    "assurance program.",
    "CRMS (Customer Relationship Management System) holds customer-facing "
    "data such as subscriptions, plans, and account status, while RMS "
    "(Rating and Mediation System) processes raw usage into rated, "
    "billable events. Discrepancies arise when a plan change in CRMS is "
    "not reflected in RMS in time, or vice versa — a classic example of "
    "leakage caused by system desynchronization rather than a single "
    "faulty component.",
    "A typical telecom ETL pipeline supporting revenue assurance has four "
    "stages: Extract (pull CDRs and reference data from source systems), "
    "Validate (check completeness, duplicates, and schema conformity), "
    "Transform (apply rating rules, currency conversion, and business "
    "logic), and Load (persist into the reconciliation or billing "
    "warehouse for reporting and KPI tracking.",
    "Common reconciliation discrepancy patterns include duplicate CDRs "
    "counted twice, rating mismatches where the wrong tariff was applied, "
    "timing gaps where usage falls on either side of a billing cycle "
    "boundary, and dropped records lost during mediation due to system "
    "downtime or format errors. Industry frameworks catalog dozens of "
    "such recurring leakage points across order management, provisioning, "
    "and billing.",
    "CPI (Configuration and Pricing Item) validation checks that pricing "
    "plans, discount rules, and product configurations are correctly and "
    "consistently deployed across every downstream system — CRMS, RMS, "
    "and the billing engine — before they go live. A misconfigured CPI "
    "can silently cause leakage across every customer on that plan, which "
    "is why validation is treated as a preventive control rather than a "
    "detective one.",
    "Rating and mediation is the stage where raw network usage is "
    "converted into a priced, billable event by applying tariff rules. "
    "Because it sits directly between the network and the billing system, "
    "it is one of the most common points of origin for revenue leakage — "
    "a single misapplied rate can affect thousands of records before "
    "detection.",
    "Revenue Assurance practice has evolved through stages of maturity: "
    "from leakage detection and recovery, to leakage prevention, to "
    "broader risk mitigation across the business — a progression formally "
    "tracked by maturity models published by TM Forum. Key KPIs include "
    "leakage percentage (revenue lost as a share of total expected "
    "revenue), reconciliation match rate (records that reconcile cleanly "
    "without manual intervention), and discrepancy aging (how long "
    "unresolved mismatches remain open).",
]
