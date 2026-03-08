"""TEI XML parser backed by BeautifulSoup."""

from __future__ import annotations

from collections.abc import Iterable

from bs4 import BeautifulSoup
from bs4.element import PageElement, Tag

from doc_visualizer.phase1.models import DocumentSections, PaperMetadata, ParsedDocument, RawSection
from doc_visualizer.phase1.section_classifier import classify_section_heading
from doc_visualizer.phase1.text_utils import (
    extract_first_year,
    join_non_empty,
    merge_text,
    normalize_whitespace,
)
from doc_visualizer.phase1.types import SectionName, empty_section_map


class BeautifulSoupTeiParser:
    """Parse GROBID TEI XML and isolate content sections from metadata."""

    def parse(self, tei_xml: str) -> ParsedDocument:
        """Convert TEI XML payload into strongly typed section + metadata objects."""
        soup = _build_soup(tei_xml)
        _strip_citation_refs(soup)

        sections = _extract_sections(soup)
        metadata = _extract_metadata(soup)
        raw_sections = _extract_raw_sections(soup)

        return ParsedDocument(
            sections=DocumentSections.from_mapping(sections),
            metadata=metadata,
            raw_sections=raw_sections,
        )


def _build_soup(tei_xml: str) -> BeautifulSoup:
    """Build a BeautifulSoup XML tree."""
    return BeautifulSoup(tei_xml, "xml")


def _strip_citation_refs(soup: BeautifulSoup) -> None:
    """Remove bibliography reference markers from textual content."""
    for ref_tag in _iter_tags(soup.find_all("ref", attrs={"type": "bibr"})):
        ref_tag.decompose()


def _extract_sections(soup: BeautifulSoup) -> dict[SectionName, str]:
    """Extract and normalize core scientific sections."""
    sections = empty_section_map()

    abstract_text = _tag_text(soup.select_one("abstract"))
    if abstract_text:
        sections["abstract"] = abstract_text

    for body_div in _iter_tags(soup.select("text > body div")):
        heading_text = _tag_text(body_div.find("head"))
        section_name = classify_section_heading(heading_text)
        if section_name is None:
            continue

        section_text = _extract_div_text(body_div, heading_text)
        if not section_text:
            continue

        sections[section_name] = merge_text(sections[section_name], section_text)

    return sections


def _extract_div_text(div_tag: Tag, heading_text: str, recursive: bool = True) -> str:
    """Extract paragraph text from a body <div> while excluding heading text."""
    paragraphs: list[str] = []
    for paragraph in _iter_tags(div_tag.find_all("p", recursive=recursive)):
        paragraph_text = _tag_text(paragraph)
        if paragraph_text:
            paragraphs.append(paragraph_text)

    merged_paragraphs = join_non_empty(paragraphs)
    if merged_paragraphs:
        return merged_paragraphs

    div_text = _tag_text(div_tag)
    if not div_text:
        return ""

    if heading_text:
        lower_text = div_text.lower()
        lower_heading = heading_text.lower()
        if lower_text.startswith(lower_heading):
            return normalize_whitespace(div_text[len(heading_text) :])

    return div_text


def _extract_raw_sections(soup: BeautifulSoup) -> tuple[RawSection, ...]:
    """Extract all body div sections, including non-standard/subsection headings."""
    body_tag = soup.select_one("text > body")
    if not isinstance(body_tag, Tag):
        return ()

    raw_sections: list[RawSection] = []
    position = 1
    for div_tag in _iter_tags(body_tag.find_all("div")):
        heading_text = _resolve_raw_section_title(div_tag=div_tag, position=position)
        section_text = _extract_div_text(
            div_tag=div_tag,
            heading_text=heading_text,
            recursive=False,
        )
        if not section_text:
            continue

        raw_sections.append(
            RawSection(
                title=heading_text,
                text=section_text,
                level=_div_level(div_tag),
                position=position,
            )
        )
        position += 1

    return tuple(raw_sections)


def _resolve_raw_section_title(div_tag: Tag, position: int) -> str:
    """Resolve a stable title for a raw section node."""
    explicit_title = _tag_text(div_tag.find("head", recursive=False))
    if explicit_title:
        return explicit_title

    type_hint = normalize_whitespace(str(div_tag.get("type", "")))
    n_hint = normalize_whitespace(str(div_tag.get("n", "")))
    combined_hint = join_non_empty((type_hint, n_hint))
    if combined_hint:
        return combined_hint

    return f"untitled_section_{position}"


def _div_level(div_tag: Tag) -> int:
    """Return div nesting level relative to body (1 = top-level section)."""
    level = 1
    parent = div_tag.parent
    while isinstance(parent, Tag):
        if parent.name == "div":
            level += 1
        if parent.name == "body":
            break
        parent = parent.parent
    return level


def _extract_metadata(soup: BeautifulSoup) -> PaperMetadata:
    """Extract bibliographic metadata from TEI."""
    title = _extract_title(soup)
    authors = _extract_authors(soup)
    year = _extract_year(soup)
    references = _extract_references(soup)

    return PaperMetadata(
        title=title,
        authors=authors,
        year=year,
        references=references,
    )


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract document title from likely TEI locations."""
    title_candidates = (
        soup.select_one("titleStmt > title"),
        soup.select_one("analytic > title"),
        soup.select_one("teiHeader title"),
        soup.find("title"),
    )

    for candidate in title_candidates:
        text = _tag_text(candidate)
        if text:
            return text

    return ""


def _extract_authors(soup: BeautifulSoup) -> tuple[str, ...]:
    """Extract author names preserving source order."""
    seen: set[str] = set()
    ordered_authors: list[str] = []

    author_tags = list(_iter_tags(soup.select("sourceDesc author")))
    if not author_tags:
        author_tags = list(_iter_tags(soup.select("analytic author")))

    for author_tag in author_tags:
        name = _author_name(author_tag)
        if not name or name in seen:
            continue
        ordered_authors.append(name)
        seen.add(name)

    return tuple(ordered_authors)


def _author_name(author_tag: Tag) -> str:
    """Extract one author's full name."""
    pers_name_tag = author_tag.find("persName")
    if pers_name_tag is None:
        return _tag_text(author_tag)

    forenames: list[str] = []
    for tag in _iter_tags(pers_name_tag.find_all("forename")):
        text = _tag_text(tag)
        if text:
            forenames.append(text)

    surname = _tag_text(pers_name_tag.find("surname"))

    if not forenames and not surname:
        return _tag_text(pers_name_tag)

    name_chunks = [*forenames, surname]
    return join_non_empty(name_chunks)


def _extract_year(soup: BeautifulSoup) -> int | None:
    """Extract publication year from prioritized date fields."""
    date_candidates: list[Tag | None] = [
        soup.select_one("publicationStmt date"),
        soup.select_one("sourceDesc date"),
        *list(_iter_tags(soup.find_all("date"))),
    ]

    for date_tag in date_candidates:
        if date_tag is None:
            continue

        when_value = str(date_tag.get("when", ""))
        tag_text = _tag_text(date_tag)
        year = extract_first_year(join_non_empty((when_value, tag_text)))
        if year is not None:
            return year

    return None


def _extract_references(soup: BeautifulSoup) -> tuple[str, ...]:
    """Extract formatted references for metadata dashboard display."""
    references: list[str] = []

    for bibl_struct in _iter_tags(soup.select("listBibl > biblStruct")):
        reference_text = _format_bibl_struct(bibl_struct)
        if reference_text:
            references.append(reference_text)

    if references:
        return tuple(dict.fromkeys(references))

    for bibl_tag in _iter_tags(soup.select("listBibl > bibl")):
        reference_text = _tag_text(bibl_tag)
        if reference_text:
            references.append(reference_text)

    return tuple(dict.fromkeys(references))


def _format_bibl_struct(bibl_struct: Tag) -> str:
    """Format one <biblStruct> into a compact readable citation string."""
    title = _tag_text(bibl_struct.select_one("analytic > title"))
    if not title:
        title = _tag_text(bibl_struct.select_one("monogr > title"))

    author_names: list[str] = []
    for author_tag in _iter_tags(bibl_struct.select("analytic author")):
        author_name = _author_name(author_tag)
        if author_name:
            author_names.append(author_name)

    lead_author = author_names[0] if author_names else ""

    date_tag = bibl_struct.select_one("imprint > date")
    year: int | None = None
    if date_tag is not None:
        combined_date_text = join_non_empty((str(date_tag.get("when", "")), _tag_text(date_tag)))
        year = extract_first_year(combined_date_text)

    parts: list[str] = []
    if lead_author:
        parts.append(lead_author)
    if year is not None:
        parts.append(f"({year})")
    if title:
        parts.append(title)

    if parts:
        return normalize_whitespace(" ".join(parts))

    return _tag_text(bibl_struct)


def _tag_text(tag: PageElement | None) -> str:
    """Normalize text extracted from an XML tag object."""
    if not isinstance(tag, Tag):
        return ""
    return normalize_whitespace(tag.get_text(" ", strip=True))


def _iter_tags(raw_items: Iterable[PageElement]) -> Iterable[Tag]:
    """Iterate only over tag elements from a BeautifulSoup query result."""
    return (item for item in raw_items if isinstance(item, Tag))
