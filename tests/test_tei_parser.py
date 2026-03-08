from __future__ import annotations

from doc_visualizer.phase1.tei_parser import BeautifulSoupTeiParser

TEI_XML = """
<TEI>
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title>Example Paper on Proteins</title>
      </titleStmt>
      <publicationStmt>
        <date when="2024-05-02" />
      </publicationStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename>Alice</forename>
                <surname>Smith</surname>
              </persName>
            </author>
            <author>
              <persName>
                <forename>Bob</forename>
                <surname>Jones</surname>
              </persName>
            </author>
            <title>Example Paper on Proteins</title>
          </analytic>
          <monogr>
            <imprint>
              <date when="2024" />
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
  <text>
    <front>
      <abstract>
        <p>
          We test an extraction pipeline with <ref type="bibr" target="#r1">[1]</ref>.
        </p>
      </abstract>
    </front>
    <body>
      <div>
        <head>Introduction</head>
        <p>Intro paragraph one.</p>
        <p>Intro paragraph two <ref type="bibr" target="#r2">[2]</ref>.</p>
      </div>
      <div>
        <head>Materials and Methods</head>
        <p>Methods paragraph.</p>
      </div>
      <div>
        <head>Defining terms</head>
        <p>Glossary and operational definitions.</p>
      </div>
      <div>
        <head>Chapter 1</head>
        <p>Overview chapter content.</p>
        <div>
          <head>Types and recommendations</head>
          <p>Subtype guidance text.</p>
        </div>
      </div>
      <div>
        <head>Results</head>
        <p>Results paragraph one.</p>
      </div>
      <div>
        <head>Conclusions</head>
        <p>Conclusion paragraph.</p>
      </div>
    </body>
    <back>
      <listBibl>
        <biblStruct xml:id="r1">
          <analytic>
            <title>First Reference</title>
            <author>
              <persName>
                <forename>Carol</forename>
                <surname>Ng</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <imprint>
              <date when="2019" />
            </imprint>
          </monogr>
        </biblStruct>
        <biblStruct xml:id="r2">
          <analytic>
            <title>Second Reference</title>
            <author>
              <persName>
                <forename>Dan</forename>
                <surname>Ray</surname>
              </persName>
            </author>
          </analytic>
          <monogr>
            <imprint>
              <date when="2021" />
            </imprint>
          </monogr>
        </biblStruct>
      </listBibl>
    </back>
  </text>
</TEI>
""".strip()


def test_parser_extracts_sections_and_isolates_metadata() -> None:
    parser = BeautifulSoupTeiParser()

    parsed_document = parser.parse(TEI_XML)

    sections = parsed_document.sections.to_mapping()
    assert sections["abstract"] == "We test an extraction pipeline with ."
    assert sections["introduction"] == "Intro paragraph one. Intro paragraph two ."
    assert sections["methods"] == "Methods paragraph."
    assert sections["results"] == "Results paragraph one."
    assert sections["conclusion"] == "Conclusion paragraph."

    metadata = parsed_document.metadata
    assert metadata.title == "Example Paper on Proteins"
    assert metadata.authors == ("Alice Smith", "Bob Jones")
    assert metadata.year == 2024
    assert metadata.references == (
        "Carol Ng (2019) First Reference",
        "Dan Ray (2021) Second Reference",
    )


def test_parser_merges_repeated_sections() -> None:
    parser = BeautifulSoupTeiParser()
    tei_xml = """
    <TEI>
      <text>
        <body>
          <div>
            <head>Results</head>
            <p>First result.</p>
          </div>
          <div>
            <head>Results</head>
            <p>Second result.</p>
          </div>
        </body>
      </text>
    </TEI>
    """

    parsed_document = parser.parse(tei_xml)

    assert parsed_document.sections.results == "First result.\n\nSecond result."


def test_parser_extracts_non_standard_raw_sections() -> None:
    parser = BeautifulSoupTeiParser()

    parsed_document = parser.parse(TEI_XML)
    raw_sections = parsed_document.raw_content_payload()

    titles = [section["title"] for section in raw_sections]
    assert "Introduction" in titles
    assert "Defining terms" in titles
    assert "Chapter 1" in titles
    assert "Types and recommendations" in titles

    nested_section = next(
        section for section in raw_sections if section["title"] == "Types and recommendations"
    )
    assert nested_section["level"] == 2
    assert nested_section["text"] == "Subtype guidance text."


def test_parser_falls_back_to_empty_metadata_when_missing() -> None:
    parser = BeautifulSoupTeiParser()
    tei_xml = """
    <TEI>
      <text>
        <body>
          <div>
            <head>Introduction</head>
            <p>Intro only.</p>
          </div>
        </body>
      </text>
    </TEI>
    """

    parsed_document = parser.parse(tei_xml)

    assert parsed_document.metadata.title == ""
    assert parsed_document.metadata.authors == ()
    assert parsed_document.metadata.year is None
    assert parsed_document.metadata.references == ()
    assert parsed_document.content_payload() == {"introduction": "Intro only."}
