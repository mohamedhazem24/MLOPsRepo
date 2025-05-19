from typing import Optional
from pydantic import BaseModel
class InferenceRequest(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: Optional[float]  # Age can be missing
    SibSp: int
    Parch: int
    Ticket: str
    Fare: Optional[float]  # Fare can also be missing
    Cabin: Optional[str]   # Cabin is often missing
    Embarked: Optional[str]  # Embarked can be missing too
