from __future__ import annotations

import pytest

from predictor import media
from predictor.config import LEAGUES_BY_ID

PREMIER_LEAGUE = LEAGUES_BY_ID["premier-league"]


def _team(name: str, **overrides: str) -> dict:
    team = {
        "idTeam": "1",
        "strTeam": name,
        "strSport": "Soccer",
        "strGender": "Male",
        "strCountry": "England",
        "strLeague": "English Premier League",
    }
    team.update(overrides)
    return team


def test_pick_team_rejects_namesakes() -> None:
    netball = _team("Nottingham Forest", strSport="Netball", strLeague="UK Netball Superleague")
    women = _team("AFC Bournemouth Women", strGender="Female", strLeague="FA Womens Challenge Cup")
    assert media._pick_team([netball], PREMIER_LEAGUE) is None
    assert media._pick_team([women], PREMIER_LEAGUE) is None


def test_pick_team_prefers_the_league_match() -> None:
    other = _team("Arsenal", idTeam="9", strLeague="English League Championship")
    right = _team("Arsenal", idTeam="133604")
    assert media._pick_team([other, right], PREMIER_LEAGUE)["idTeam"] == "133604"


class _Response:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code = status
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise media.requests.HTTPError(str(self.status_code))

    def json(self) -> dict:
        return self._payload


class _Session:
    def __init__(self, statuses: list[int]) -> None:
        self.statuses = statuses
        self.calls = 0

    def get(self, *args: object, **kwargs: object) -> _Response:
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        return _Response(status, {"teams": []})


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media.time, "sleep", lambda _: None)


def test_client_retries_then_succeeds() -> None:
    client = media.SportsDb()
    client._session = _Session([429, 429, 200])
    assert client.get("searchteams.php", t="Arsenal") == {"teams": []}
    assert client.available


def test_client_gives_up_after_repeated_refusals() -> None:
    client = media.SportsDb()
    session = _Session([429])
    client._session = session
    for _ in range(media.SportsDb.MAX_STRIKES):
        with pytest.raises(media.MediaUnavailable):
            client.get("searchteams.php", t="Arsenal")
    assert not client.available

    calls = session.calls
    with pytest.raises(media.MediaUnavailable):
        client.get("searchteams.php", t="Arsenal")
    assert session.calls == calls, "no request should be sent once the client has given up"


def test_pick_team_requires_the_searched_name() -> None:
    reserve = _team("Deportivo Fabril", strCountry="Spain", strLeague="Spanish Segunda Federacion")
    unrelated = _team("Torcy", strCountry="France", strLeague="French National 3")
    real = _team("Deportivo La Coruña", strCountry="Spain", strLeague="Spanish La Liga")
    la_liga = LEAGUES_BY_ID["la-liga"]
    assert media._pick_team([reserve], la_liga, "Deportivo La Coruna") is None
    assert media._pick_team([unrelated], LEAGUES_BY_ID["ligue-1"], "Paris SG") is None
    assert media._pick_team([real], la_liga, "Deportivo La Coruna") is real


def test_upload_time_is_read_from_the_file_name() -> None:
    url = "https://r2.thesportsdb.com/images/media/team/fanart/tuuwpy1424110617.jpg"
    assert media._uploaded_at(url) == 1424110617
    assert media._uploaded_at("https://example.com/no-stamp.jpg") is None


def test_recent_uploads_are_never_treated_as_photos() -> None:
    # 2024-11 upload: the era of league-branded wallpapers. Rejected without any download.
    url = "https://r2.thesportsdb.com/images/media/team/fanart/v2gwen1731827710.jpg"
    assert not media.is_photograph(url, session=None)  # type: ignore[arg-type]


def test_flat_graphics_are_told_apart_from_photographs() -> None:
    import io

    import numpy as np
    from PIL import Image

    def png(array: np.ndarray) -> bytes:
        buffer = io.BytesIO()
        Image.fromarray(array.astype("uint8")).save(buffer, format="PNG")
        return buffer.getvalue()

    flat = np.zeros((90, 160, 3))
    flat[:, :80] = (20, 40, 160)
    flat[30:60, 60:100] = (255, 255, 255)
    noisy = np.random.default_rng(0).integers(0, 256, size=(90, 160, 3))
    assert media._flat_share(png(flat)) > media.FLAT_SHARE_LIMIT
    assert media._flat_share(png(noisy)) < media.FLAT_SHARE_LIMIT
