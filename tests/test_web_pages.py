from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path


WEB = Path(__file__).parents[1] / "web"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.body_page = ""
        self.views: list[str] = []
        self.nav_targets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "body":
            self.body_page = values.get("data-page", "")
        if tag == "section":
            classes = set((values.get("class") or "").split())
            if "view" in classes:
                self.views.append(values.get("data-page", ""))
        if tag == "a" and "nav-item" in set((values.get("class") or "").split()):
            href = values.get("href", "")
            if href.startswith("/web/") and href.endswith(".html"):
                self.nav_targets.append(href.removeprefix("/web/"))


def parse(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8-sig"))
    return parser


def test_every_web_page_has_one_matching_view() -> None:
    pages = sorted(WEB.glob("*.html"))
    assert len(pages) == 21
    for path in pages:
        parser = parse(path)
        assert parser.body_page, path.name
        assert parser.views == [parser.body_page], path.name


def test_navigation_targets_are_complete_and_local() -> None:
    pages = {path.name for path in WEB.glob("*.html")}
    parser = parse(WEB / "index.html")
    assert set(parser.nav_targets) == pages
    assert len(parser.nav_targets) == len(set(parser.nav_targets))
    for path in WEB.glob("*.html"):
        assert set(parse(path).nav_targets) == pages


def test_index_is_dashboard_only() -> None:
    parser = parse(WEB / "index.html")
    assert parser.body_page == "dashboard"
    assert parser.views == ["dashboard"]
