# Extended Mock Datasets for deeper operational context

ACCOUNTS_DB = {
    "CUST-1001": {
        "name": "Jane Smith",
        "tenure_years": 3.5,
        "lifetime_value": 450.00,
        "vip_status": False,
        "fraud_flag": False,
        "churn_risk": "low"
    },
    "CUST-1002": {
        "name": "John Doe",
        "tenure_years": 0.5,
        "lifetime_value": 85.00,
        "vip_status": False,
        "fraud_flag": False,
        "churn_risk": "medium"
    },
    "CUST-2099": {
        "name": "Alice Fraudster",
        "tenure_years": 0.1,
        "lifetime_value": 0.00,
        "vip_status": False,
        "fraud_flag": True,
        "churn_risk": "high"
    }
}

ORDERS_DB = {
    "ORD-1A": {
        "customer_id": "CUST-1001",
        "items": ["Ceramic Coffee Mug"],
        "total_cost": 24.99,
        "status": "delivered",
        "delivery_date": "2024-02-10",
        "shipping_address": "123 Elm St, NY",
        "sla_severity": "low"
    },
    "ORD-2B": {
        "customer_id": "CUST-1002",
        "items": ["Wireless Earbuds"],
        "total_cost": 49.99,
        "status": "shipped_delayed",
        "delivery_date": "Pending",
        "shipping_address": "456 Oak Ave, CA",
        "sla_severity": "medium"
    },
    "ORD-3C": {
        "customer_id": "CUST-2099",
        "items": ["Luxury Watch"],
        "total_cost": 899.99,
        "status": "processing",
        "delivery_date": "Pending",
        "shipping_address": "PO BOX 999, NV",
        "sla_severity": "high"
    }
}

REFUND_HISTORY_DB = {
    "CUST-1001": {"previous_refunds": 0, "total_refund_amount": 0.00},
    "CUST-1002": {"previous_refunds": 1, "total_refund_amount": 12.00, "reason": "item_not_fitting"},
    "CUST-2099": {"previous_refunds": 4, "total_refund_amount": 3200.00, "reason": "multiple_missing_items"}
}

BILLING_HISTORY_DB = {
    "CUST-1001": {"payment_method": "Credit Card ending in 1234", "status": "settled"},
    "CUST-1002": {"payment_method": "Paypal", "status": "settled"},
    "CUST-2099": {"payment_method": "Multiple Virtual Cards", "status": "chargeback_warning"}
}

RETURN_POLICIES = {
    "general": "Items can be returned within 30 days of delivery. Damaged items receive full refund. Customer keeps the damaged item.",
    "delayed_shipping": "If an item is delayed beyond 7 days of promised delivery, we offer a 15% courtesy refund or store credit. No full cancellations unless requested specifically after 14 days delay.",
    "fraud_protocol": "If fraud_flag is True OR previous refunds exceed 2, all refunds must be denied. The ticket must be escalated to billing_ops. Do NOT issue refund under any circumstances or offer replacement."
}
