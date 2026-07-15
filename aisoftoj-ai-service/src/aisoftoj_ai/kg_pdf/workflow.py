from typing import Any

from aisoftoj_ai.kg_pdf.kg_document_parser import (
    infer_document_title,
    parse_content_list,
    parse_markdown,
)
from aisoftoj_ai.kg_pdf.kg_extraction_chunks import build_kg_extraction_chunks
from aisoftoj_ai.kg_pdf.kg_relation_extractor import (
    DEFAULT_CHUNK_TIMEOUT_SECONDS,
    DEFAULT_EXTRACTION_CONCURRENCY,
    extract_kg_relations,
)
from aisoftoj_ai.kg_pdf.kg_structure_builder import build_document_structure, flatten_heading_nodes
from aisoftoj_ai.kg_pdf.models import KgExtractionResult


async def run_kg_pdf_extraction(
    chat,
    document_id: str,
    version: int = 1,
    content_list: list[dict[str, Any]] | None = None,
    markdown: str = "",
    document_title: str | None = None,
    chunk_batch_size: int = 72,
    extraction_concurrency: int = DEFAULT_EXTRACTION_CONCURRENCY,
    chunk_timeout_seconds: float = DEFAULT_CHUNK_TIMEOUT_SECONDS,
) -> KgExtractionResult:
    blocks = parse_content_list(content_list or [])
    if not blocks and markdown.strip():
        blocks = parse_markdown(markdown)
    if not blocks:
        raise RuntimeError("Document has no raw parse blocks for KG extraction")

    title = infer_document_title(document_id, blocks, document_title)
    structure, bindings = build_document_structure(document_id, title, blocks)
    chunks = build_kg_extraction_chunks(document_id, bindings, version=version)
    entities, relations = await extract_kg_relations(
        chat,
        structure,
        chunks,
        chunk_batch_size,
        document_version=version,
        concurrency=extraction_concurrency,
        chunk_timeout_seconds=chunk_timeout_seconds,
    )
    heading_count = len(flatten_heading_nodes(structure.nodes))

    return KgExtractionResult(
        document_id=document_id,
        version=version,
        document_structure=structure,
        kg_extraction_chunks=chunks,
        entity_nodes=entities,
        relations=relations,
        stats={
            "sourceBlockCount": len(blocks),
            "headingNodeCount": heading_count,
            "kgChunkCount": len(chunks),
            "entityCount": len(entities),
            "relationCount": len(relations),
        },
    )
