import json
import math
import re
import unicodedata
from typing import Any

ALIGNMENT_SYSTEM_PROMPT = """
从每道错题自己的候选列表中选择最相关的文档知识点，只返回符合 schema 的 JSON。
规则：
1. knowledge_point_id 必须来自该题的 candidate_knowledge_point_ids，不能编造。
2. 没有把握时不要输出该题映射。
3. 优先依据题干和解析，再用标题路径和原文摘录消歧，不能只按同名匹配。
4. 同名知识点位于不同标题路径时，要根据语义消歧。
5. 只输出 confidence >= 0.60 的映射。
"""

ALIGNMENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "wrong_question_kg_alignment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "alignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "question_id": {"type": "string"},
                            "knowledge_point_id": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "reason": {"type": "string"},
                            "evidence_text": {"type": "string"},
                            "context_dependency": {
                                "type": "string",
                                "enum": [
                                    "question_text",
                                    "analysis",
                                    "heading_path",
                                    "chunk_context",
                                ],
                            },
                        },
                        "required": [
                            "question_id",
                            "knowledge_point_id",
                            "confidence",
                            "reason",
                            "evidence_text",
                            "context_dependency",
                        ],
                    },
                }
            },
            "required": ["alignments"],
        },
    },
}


async def align_wrong_questions_to_kg(
    chat,
    wrong_questions: list[dict[str, Any]],
    entity_nodes: list[dict[str, Any]],
    kg_extraction_chunks: list[dict[str, Any]],
    max_alignments: int = 120,
    question_batch_size: int = 4,
    candidates_per_question: int = 8,
) -> dict[str, Any]:
    """Map wrong questions with local lexical recall plus Qwen judgement.

    The lexical stage is deterministic and has no embedding/reranker dependency.
    Qwen only sees a small candidate set for each question, which keeps prompts
    bounded for large document graphs.
    """
    candidates = _build_candidates(entity_nodes, kg_extraction_chunks)
    questions = _build_questions(wrong_questions)
    if not chat or not candidates or not questions:
        return {"alignments": []}

    candidate_by_id = {item["knowledge_point_id"]: item for item in candidates}
    collected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for start in range(0, len(questions), max(1, question_batch_size)):
        if len(collected) >= max_alignments:
            break
        batch = questions[start : start + max(1, question_batch_size)]
        allowed_by_question: dict[str, set[str]] = {}
        batch_candidates: dict[str, dict[str, Any]] = {}
        prompt_questions: list[dict[str, Any]] = []

        for question in batch:
            ranked = _rank_candidates(question, candidates, candidates_per_question)
            allowed_ids = {
                candidate["knowledge_point_id"] for candidate in ranked
            }
            allowed_by_question[question["question_id"]] = allowed_ids
            for candidate_id in allowed_ids:
                batch_candidates[candidate_id] = candidate_by_id[candidate_id]
            prompt_questions.append(
                {
                    **question,
                    "candidate_knowledge_point_ids": sorted(allowed_ids),
                }
            )

        if not batch_candidates:
            continue
        raw = await chat.complete(
            [
                {"role": "system", "content": ALIGNMENT_SYSTEM_PROMPT.strip()},
                {
                    "role": "user",
                    "content": _build_prompt(
                        prompt_questions,
                        list(batch_candidates.values()),
                        max_alignments - len(collected),
                    ),
                },
            ],
            temperature=0.02,
            response_format=ALIGNMENT_RESPONSE_FORMAT,
        )
        parsed = _extract_json(raw)
        sanitized = _sanitize_alignments(
            parsed,
            prompt_questions,
            list(batch_candidates.values()),
            max_alignments - len(collected),
            allowed_by_question,
        )

        for alignment in sanitized["alignments"]:
            key = (alignment["question_id"], alignment["knowledge_point_id"])
            if key not in seen:
                seen.add(key)
                collected.append(alignment)

    return {"alignments": collected[:max_alignments]}


def _build_candidates(
    entity_nodes: list[dict[str, Any]],
    kg_extraction_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunks_by_id = {
        _text(chunk.get("kg_chunk_id")): chunk
        for chunk in kg_extraction_chunks
        if _text(chunk.get("kg_chunk_id"))
    }
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entity in entity_nodes:
        entity_id = _text(entity.get("entity_id"))
        name = _text(entity.get("canonical_name")) or _text(entity.get("name"))
        if not entity_id or not name or entity_id in seen:
            continue
        seen.add(entity_id)
        source_chunks = []
        for chunk_id in _string_list(entity.get("source_kg_chunk_ids"))[:2]:
            chunk = chunks_by_id.get(chunk_id)
            if not chunk:
                continue
            source_chunks.append(
                {
                    "kg_chunk_id": chunk_id,
                    "source_page_range": _text(chunk.get("source_page_range")),
                    "heading_path": _string_list(chunk.get("heading_path")),
                    "text": _truncate(_text(chunk.get("text")), 300),
                }
            )
        candidates.append(
            {
                "knowledge_point_id": entity_id,
                "name": name,
                "aliases": _string_list(entity.get("aliases")),
                "heading_path": _string_list(entity.get("heading_path")),
                "disambiguation_key": _text(entity.get("disambiguation_key")),
                "source_chunks": source_chunks,
            }
        )
    return candidates[:1000]


def _build_questions(wrong_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in wrong_questions:
        question_id = _text(raw.get("questionId") or raw.get("question_id"))
        if not question_id or question_id in seen:
            continue
        seen.add(question_id)
        questions.append(
            {
                "question_id": question_id,
                "question_name": _text(raw.get("questionName") or raw.get("question_name")),
                "question_knowledge_hint": _text(
                    raw.get("knowledgePointName") or raw.get("knowledge_point_name")
                ),
                "subject": _text(raw.get("subjectName") or raw.get("subject_name")),
                "question_intro": _truncate(
                    _text(raw.get("questionIntro") or raw.get("question_intro")), 600
                ),
                "options": _truncate(_text(raw.get("options")), 400),
                "analysis": _truncate(_text(raw.get("analysis")), 600),
            }
        )
    return questions[:120]


def _rank_candidates(
    question: dict[str, Any],
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    question_text = " ".join(
        _text(question.get(key))
        for key in (
            "question_name",
            "question_knowledge_hint",
            "subject",
            "question_intro",
            "options",
            "analysis",
        )
    )
    question_terms = _lexical_terms(question_text)
    normalized_question = _normalize_text(question_text)
    knowledge_hint = _normalize_text(_text(question.get("question_knowledge_hint")))

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for candidate in candidates:
        name = _normalize_text(_text(candidate.get("name")))
        candidate_text = " ".join(
            [
                _text(candidate.get("name")),
                " ".join(_string_list(candidate.get("aliases"))),
                " ".join(_string_list(candidate.get("heading_path"))),
                " ".join(
                    _text(chunk.get("text"))
                    for chunk in candidate.get("source_chunks", [])
                ),
            ]
        )
        candidate_terms = _lexical_terms(candidate_text)
        overlap = len(question_terms & candidate_terms)
        denominator = math.sqrt(max(1, len(question_terms) * len(candidate_terms)))
        score = overlap / denominator
        if name and name in normalized_question:
            score += 4.0
        if knowledge_hint and (
            knowledge_hint == name or knowledge_hint in name or name in knowledge_hint
        ):
            score += 6.0
        scored.append((score, candidate["knowledge_point_id"], candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored if item[0] > 0][: max(1, limit)]


def _lexical_terms(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    terms = set(re.findall(r"[a-z0-9][a-z0-9_+#.-]{1,}|[\u4e00-\u9fff]{2,}", normalized))
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", normalized):
        for size in (2, 3):
            terms.update(
                sequence[index : index + size]
                for index in range(max(0, len(sequence) - size + 1))
            )
    return terms


def _build_prompt(
    questions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    max_alignments: int,
) -> str:
    payload = {
        "wrong_questions": questions,
        "candidate_knowledge_points": candidates,
        "limits": {"max_alignments": max_alignments},
    }
    return (
        "请逐题在该题的 candidate_knowledge_point_ids 范围内判断最匹配的文档知识点。\n"
        "只保留证据充分且 confidence >= 0.60 的映射。\n"
        "必须结合题干、解析、标题路径和 chunk 上下文进行消歧。\n\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )


def _sanitize_alignments(
    parsed: dict[str, Any],
    questions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    max_alignments: int,
    allowed_by_question: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    question_ids = {_text(item.get("question_id")) for item in questions}
    candidate_ids = {_text(item.get("knowledge_point_id")) for item in candidates}
    alignments: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in parsed.get("alignments", []):
        question_id = _text(raw.get("question_id") or raw.get("questionId"))
        knowledge_point_id = _text(
            raw.get("knowledge_point_id")
            or raw.get("knowledgePointId")
            or raw.get("entity_id")
            or raw.get("entityId")
        )
        confidence = _clamp_float(raw.get("confidence"), 0.0, 1.0, 0.0)
        key = (question_id, knowledge_point_id)
        allowed = not allowed_by_question or knowledge_point_id in allowed_by_question.get(
            question_id, set()
        )
        if (
            question_id not in question_ids
            or knowledge_point_id not in candidate_ids
            or not allowed
            or confidence < 0.60
            or key in seen
        ):
            continue
        seen.add(key)
        alignments.append(
            {
                "question_id": question_id,
                "knowledge_point_id": knowledge_point_id,
                "confidence": confidence,
                "reason": _truncate(_text(raw.get("reason")), 220),
                "evidence_text": _truncate(
                    _text(raw.get("evidence_text") or raw.get("evidence")), 220
                ),
                "context_dependency": _text(raw.get("context_dependency"))
                or "chunk_context",
                "mapping_method": "qwen_lexical_alignment",
            }
        )
        if len(alignments) >= max_alignments:
            break
    return {"alignments": alignments}


def _extract_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    text = _text(value)
    return [text] if text else []


def _normalize_text(value: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffa-z0-9]+", "", unicodedata.normalize("NFKC", value).lower())


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit]


def _clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))
