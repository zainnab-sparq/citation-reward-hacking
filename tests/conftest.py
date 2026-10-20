"""Shared test fixtures."""
import httpx
import pytest
import respx


@pytest.fixture
def mock_title_searches_empty():
    """Default-mock the three title-search endpoints to return no matches.

    The title fallback path always runs when DOI lookup doesn't find a match,
    so every test that exercises the orchestrator hits these endpoints. Default
    them to definitive-empty so adding a new title source only requires updating
    the fixture, not every test.

    Tests still mock their own DOI lookup endpoints explicitly.
    """
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://api.crossref.org/works", name="crossref_title").mock(
            return_value=httpx.Response(200, json={"message": {"items": []}})
        )
        mock.get(
            "https://api.semanticscholar.org/graph/v1/paper/search", name="s2_title"
        ).mock(return_value=httpx.Response(200, json={"data": []}))
        mock.get("https://api.openalex.org/works", name="openalex_title").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        yield mock
