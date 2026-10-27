from pydantic import BaseModel


class FAISSSetupRequest(BaseModel):
    known_crops_dir: str | None = None
    class_subdirs: bool = True
