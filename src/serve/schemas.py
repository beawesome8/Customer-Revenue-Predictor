"""
Pydantic request/response schemas for the prediction API.

Fields mirror the raw training columns minus Revenue (the target) and
Month (re-encoded server-side into sin/cos, so callers send plain month
names and don't need to know about that transform).
"""
from typing import Literal
from pydantic import BaseModel, Field


class SessionFeatures(BaseModel):
    Administrative: int = Field(ge=0)
    Administrative_Duration: float = Field(ge=0)
    Informational: int = Field(ge=0)
    Informational_Duration: float = Field(ge=0)
    ProductRelated: int = Field(ge=0)
    ProductRelated_Duration: float = Field(ge=0)
    BounceRates: float = Field(ge=0, le=1)
    ExitRates: float = Field(ge=0, le=1)
    PageValues: float = Field(ge=0)
    SpecialDay: float = Field(ge=0, le=1)
    Month: Literal["Jan", "Feb", "Mar", "Apr", "May", "June",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: Literal["Returning_Visitor", "New_Visitor", "Other"]
    Weekend: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "Administrative": 2, "Administrative_Duration": 45.0,
                "Informational": 0, "Informational_Duration": 0.0,
                "ProductRelated": 15, "ProductRelated_Duration": 600.0,
                "BounceRates": 0.02, "ExitRates": 0.05, "PageValues": 12.5,
                "SpecialDay": 0.0, "Month": "Nov", "OperatingSystems": 2,
                "Browser": 2, "Region": 1, "TrafficType": 2,
                "VisitorType": "Returning_Visitor", "Weekend": False,
            }
        }
    }


class PredictionResponse(BaseModel):
    purchase_probability: float
    will_purchase: bool
    model_version: str
