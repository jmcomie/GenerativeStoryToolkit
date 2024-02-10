import shutil
import urllib.parse
from pathlib import Path
from typing import Optional

from appdirs import user_data_dir


class ProjectLocator:
    ROOT_DIRECTORY_NAME: str = "GenerativeStoryToolkit"
    PROJECTS_DIRECTORY_NAME: str = "gstk_projects"

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path(user_data_dir(ProjectLocator.ROOT_DIRECTORY_NAME))
        base_path = base_path / ProjectLocator.PROJECTS_DIRECTORY_NAME
        self._base_path = base_path

    @property
    def base_path(self) -> Path:
        return self._base_path

    @base_path.setter
    def base_path(self, value: Path):
        self._base_path = value

    def list_project_ids(self) -> list[str]:
        try:
            return [
                urllib.parse.unquote(project_path.name)
                for project_path in self._base_path.iterdir()
                if project_path.is_dir()
            ]
        except FileNotFoundError:
            return []

    def project_id_exists(self, project_id: str) -> bool:
        return (self._base_path / urllib.parse.quote(project_id, safe="")).exists()

    def get_project_resource_location(self, project_id: str) -> Path:
        return self._base_path / urllib.parse.quote(project_id, safe="")

    def delete_project(self, project_id: str) -> None:
        project_path = self.get_project_resource_location(project_id)
        if project_path.exists():
            shutil.rmtree(project_path)
