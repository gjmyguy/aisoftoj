import json

from aisoftoj_ai.kg_pdf.kg_extraction_chunks import build_kg_extraction_chunks
from aisoftoj_ai.kg_pdf.kg_relation_extractor import (
    _normalize_extraction,
    extract_kg_relations,
)
from aisoftoj_ai.kg_pdf.models import (
    KgBlockBinding,
    KgDocumentStructure,
    KgExtractionChunk,
    KgSourceBlock,
)
from aisoftoj_ai.kg_pdf.wrong_question_aligner import (
    _rank_candidates,
    align_wrong_questions_to_kg,
)


def _chunk() -> KgExtractionChunk:
    return KgExtractionChunk(
        document_id="doc-a",
        kg_chunk_id="chunk-a",
        source_page_range="12",
        page_start=12,
        page_end=12,
        heading_path=["项目管理", "风险管理"],
        parent_heading_id="heading-a",
        text="风险识别用于发现项目中的不确定因素。",
    )


def test_document_entity_ids_are_version_scoped():
    extracted = [(_chunk(), {"entities": [{"name": "风险识别"}], "relations": []})]

    version_one, _ = _normalize_extraction(extracted, "doc-a", 1)
    version_two, _ = _normalize_extraction(extracted, "doc-a", 2)
    other_document, _ = _normalize_extraction(extracted, "doc-b", 1)

    assert version_one[0].entity_id.startswith("entity:doc-a:v1:")
    assert version_one[0].entity_id != version_two[0].entity_id
    assert version_one[0].entity_id != other_document[0].entity_id


def test_alias_resolves_to_existing_entity_in_same_heading_scope():
    chunk = _chunk()
    extracted = [
        (
            chunk,
            {
                "entities": [
                    {"name": "风险辨识", "aliases": ["风险识别"]},
                    {"name": "风险识别", "aliases": []},
                ],
                "relations": [],
            },
        )
    ]

    entities, _ = _normalize_extraction(extracted, "doc-a", 1)

    assert len(entities) == 1
    assert "风险识别" in entities[0].aliases


def test_kg_chunk_ids_are_version_scoped():
    bindings = [
        KgBlockBinding(
            block=KgSourceBlock(content="风险识别用于发现不确定因素", page=3),
            parent_heading_id="heading-risk",
            heading_path=["项目管理", "风险管理"],
        )
    ]

    version_one = build_kg_extraction_chunks("doc-a", bindings, version=1)
    version_two = build_kg_extraction_chunks("doc-a", bindings, version=2)

    assert version_one[0].kg_chunk_id.startswith("kg_chunk_v1_")
    assert version_two[0].kg_chunk_id.startswith("kg_chunk_v2_")
    assert version_one[0].kg_chunk_id != version_two[0].kg_chunk_id


class _ExtractionChat:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages, temperature, response_format=None):
        self.calls += 1
        assert response_format["type"] == "json_schema"
        return json.dumps(
            {
                "entities": [{"name": f"知识点{self.calls}", "aliases": []}],
                "relations": [],
            },
            ensure_ascii=False,
        )


async def test_chunk_batch_size_does_not_truncate_document():
    chat = _ExtractionChat()
    chunks = [
        _chunk(),
        _chunk().model_copy(
            update={
                "kg_chunk_id": "chunk-b",
                "text": "风险分析用于评估风险影响。",
            }
        ),
    ]

    entities, _ = await extract_kg_relations(
        chat,
        KgDocumentStructure(document_id="doc-a", title="项目管理"),
        chunks,
        chunk_batch_size=1,
    )

    assert chat.calls == 2
    assert len(entities) == 2


def test_lexical_recall_prefers_question_semantics():
    question = {
        "question_id": "1",
        "question_name": "项目风险识别",
        "question_knowledge_hint": "风险识别",
        "question_intro": "需要发现并记录项目中的不确定因素",
        "analysis": "考查风险识别过程",
    }
    candidates = [
        {
            "knowledge_point_id": "entity:cost",
            "name": "成本估算",
            "aliases": [],
            "heading_path": ["成本管理"],
            "source_chunks": [{"text": "估算项目成本"}],
        },
        {
            "knowledge_point_id": "entity:risk",
            "name": "风险识别",
            "aliases": [],
            "heading_path": ["风险管理"],
            "source_chunks": [{"text": "识别项目中的不确定因素"}],
        },
    ]

    ranked = _rank_candidates(question, candidates, 2)

    assert ranked[0]["knowledge_point_id"] == "entity:risk"


def test_lexical_recall_does_not_inject_unrelated_candidates():
    ranked = _rank_candidates(
        {
            "question_id": "1",
            "question_name": "网络握手协议",
            "question_intro": "客户端如何建立传输连接",
        },
        [
            {
                "knowledge_point_id": "entity:cost",
                "name": "成本估算",
                "aliases": [],
                "heading_path": ["项目成本管理"],
                "source_chunks": [{"text": "估算项目预算"}],
            }
        ],
        8,
    )

    assert ranked == []


class _FakeChat:
    def __init__(self):
        self.calls = 0

    async def complete(self, messages, temperature, response_format=None):
        self.calls += 1
        assert response_format["type"] == "json_schema"
        payload = json.loads(messages[-1]["content"].split("\n\n", 1)[1])
        alignments = []
        for question in payload["wrong_questions"]:
            alignments.append(
                {
                    "question_id": question["question_id"],
                    "knowledge_point_id": question["candidate_knowledge_point_ids"][0],
                    "confidence": 0.8,
                    "reason": "题干与文档知识点一致",
                }
            )
        return json.dumps({"alignments": alignments}, ensure_ascii=False)


async def test_alignment_batches_qwen_calls_without_embedding():
    chat = _FakeChat()
    questions = [
        {
            "questionId": str(index),
            "questionName": "风险识别",
            "questionIntro": "识别项目风险",
        }
        for index in range(5)
    ]
    entities = [
        {
            "entity_id": "entity:risk",
            "name": "风险识别",
            "canonical_name": "风险识别",
            "source_kg_chunk_ids": ["chunk-a"],
        }
    ]
    chunks = [
        {
            "kg_chunk_id": "chunk-a",
            "heading_path": ["风险管理"],
            "text": "风险识别用于发现项目中的不确定因素。",
        }
    ]

    result = await align_wrong_questions_to_kg(
        chat,
        questions,
        entities,
        chunks,
        question_batch_size=2,
    )

    assert chat.calls == 3
    assert len(result["alignments"]) == 5
    assert all(
        item["mapping_method"] == "qwen_lexical_alignment"
        for item in result["alignments"]
    )
