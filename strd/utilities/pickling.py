from pathlib import Path

from data_tree.util import Pickled
from pinject_design import injected_function, Injected


@injected_function
def to_pickled(session, /, tgt: Injected, dst: Path):
    def create():
        return session[tgt]

    pkled = Pickled(str(dst), create)
    return pkled.value
