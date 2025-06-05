from dataclasses import dataclass
from typing import Callable, Optional

from abvelocity.get_data.cursor import Cursor


@dataclass
class IOParam:
    """Dataclass to hold input/output parameters."""

    cursor: Optional[Cursor] = None
    print_to_html: Optional[Callable] = None
    save_path: str = "./"
    file_name_suffix: str = ""
