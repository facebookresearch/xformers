from pathlib import Path

from xformers.components.positional_encoding.base import PositionEncoding  # noqa
from xformers.components.positional_encoding.base import PositionEncodingConfig  # noqa
from xformers.utils import import_all_modules

# automatically import any Python files in the directory
import_all_modules(
    str(Path(__file__).parent), "xformers.components.positional_encoding"
)

from xformers.components.positional_encoding.sine import SinePositionEncoding  # noqa
