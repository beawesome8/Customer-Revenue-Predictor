"""
Pure logic for the demo app, kept separate from Streamlit code so it's
actually unit-testable - Streamlit scripts run top-to-bottom on every
interaction, so importing one directly in a test runs its whole UI unless
the logic lives elsewhere.
"""
import requests

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]

# Dataset medians/modes for fields not exposed in the UI - so a demo user
# filling in just the "important" fields still gets a realistic
# prediction, not one skewed by zeroed-out features.
DEFAULT_HIDDEN_FIELDS = {
    "Administrative": 2,
    "Administrative_Duration": 80.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "SpecialDay": 0.0,
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
}


def build_payload(product_related, product_duration, bounce_rate, exit_rate,
                   page_values, month, visitor_type, weekend):
    payload = dict(DEFAULT_HIDDEN_FIELDS)
    payload.update({
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_duration,
        "BounceRates": bounce_rate,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "Month": month,
        "VisitorType": visitor_type,
        "Weekend": weekend,
    })
    return payload


def call_api(payload: dict, api_url: str, timeout: float = 10.0) -> dict:
    response = requests.post(f"{api_url}/predict", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()
