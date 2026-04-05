# Mock Datasets for simulating internal tools/databases

ACCOUNTS_DB = {
    "CUST-1001": {
        "name": "Jane Smith",
        "tenure_years": 3,
        "lifetime_value": 450.00,
        "fraud_flag": False,
        "previous_returns_count": 0
    },
    "CUST-1002": {
        "name": "John Doe",
        "tenure_years": 1,
        "lifetime_value": 25.00,
        "fraud_flag": False,
        "previous_returns_count": 0
    },
    "CUST-2099": {
        "name": "Alice Fraudster",
        "tenure_years": 0.2,
        "lifetime_value": 0.00,
        "fraud_flag": True,
        "previous_returns_count": 5
    }
}

ORDERS_DB = {
    "ORD-1A": {
        "customer_id": "CUST-1001",
        "items": ["Ceramic Coffee Mug"],
        "total_cost": 24.99,
        "status": "delivered",
        "delivery_date": "2024-02-10",
        "shipping_address": "123 Elm St, NY"
    },
    "ORD-2B": {
        "customer_id": "CUST-1002",
        "items": ["Wireless Earbuds"],
        "total_cost": 49.99,
        "status": "shipped_delayed",
        "delivery_date": "Pending",
        "shipping_address": "456 Oak Ave, CA"
    },
    "ORD-3C": {
        "customer_id": "CUST-2099",
        "items": ["Luxury Watch"],
        "total_cost": 899.99,
        "status": "processing",
        "delivery_date": "Pending",
        "shipping_address": "PO BOX 999, NV"
    }
}

RETURN_POLICIES = {
    "general": "Items can be returned within 30 days of delivery. Damaged items receive full refund. Customer keeps the damaged item.",
    "delayed_shipping": "If an item is delayed beyond 7 days of promised delivery, we offer a 15% courtesy refund or cancellation with full refund.",
    "fraud_protocol": "If fraud_flag is True, all refunds are blocked. Escalate to billing_ops immediately."
}
