from pydantic import BaseModel


class ResetRequest(BaseModel):
    delete_crops: bool = True
