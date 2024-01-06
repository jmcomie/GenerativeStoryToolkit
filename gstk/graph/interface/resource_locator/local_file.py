import urllib.parse
from pathlib import Path
from typing import Optional

from appdirs import user_data_dir

from gstk.graph.interface.resource_locator.resource_locator import ResourceLocator


class LocalFileLocator(ResourceLocator):
    ROOT_DIRECTORY_NAME: str = "GenerativeStoryToolkit"
    PROJECTS_DIRECTORY_NAME: str = "gstk_projects"

    def __init__(self, base_path: Optional[Path] = None):
        if base_path is None:
            base_path = Path(user_data_dir(LocalFileLocator.ROOT_DIRECTORY_NAME))
        base_path = base_path / LocalFileLocator.PROJECTS_DIRECTORY_NAME
        self._base_path = base_path

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
