from pydantic import BaseModel


class RollbackRequest(BaseModel):
    version_id: str
    confirmed_by: str = "dashboard"


class DeployConfirmRequest(BaseModel):
    version_id: str
