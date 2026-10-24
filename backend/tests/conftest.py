from __future__ import annotations

import pandas as pd
import pytest

from tests.factories import make_matches


@pytest.fixture
def matches() -> pd.DataFrame:
    return make_matches()
