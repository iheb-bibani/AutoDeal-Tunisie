import re

from web.routes import BY_KEY, BY_LABEL, BY_PATH, ROUTES


def test_route_registry_is_unique_and_has_one_default():
    assert len(BY_KEY) == len(ROUTES)
    assert len(BY_LABEL) == len(ROUTES)
    assert len(BY_PATH) == len(ROUTES)
    assert sum(route.default for route in ROUTES) == 1


def test_routes_are_clean_browser_paths():
    for route in ROUTES:
        assert route.url_path
        assert "/" not in route.url_path
        assert " " not in route.url_path
        assert re.fullmatch(r"[a-z0-9-]+", route.url_path), route.url_path


def test_core_public_urls_are_stable():
    expected = {
        "🏠 Accueil": "accueil",
        "🛒 Acheter": "acheter",
        "⚖️ Comparateur": "comparateur",
        "💰 Calculateur": "estimer",
        "🤖 Assistant": "assistant",
        "🤝 Samsar": "pro-samsar",
        "🏢 Concessionnaire": "pro-concessionnaire",
        "🛠️ Admin": "admin",
    }
    for label, path in expected.items():
        assert BY_LABEL[label].url_path == path
