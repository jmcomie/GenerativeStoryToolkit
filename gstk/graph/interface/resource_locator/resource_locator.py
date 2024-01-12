from abc import ABC, abstractmethod
from pathlib import Path


class ResourceLocator(ABC):
    @abstractmethod
    def list_project_ids() -> list[str]:
        pass

    @abstractmethod
    def get_project_resource_location(project_id: str) -> Path:
        pass

    @abstractmethod
    def project_id_exists(self, project_id: str) -> bool:
        pass

    @abstractmethod
    def delete_project(project_id: str) -> None:
        pass
