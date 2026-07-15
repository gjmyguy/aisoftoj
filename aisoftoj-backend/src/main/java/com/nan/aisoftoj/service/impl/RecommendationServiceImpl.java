package com.nan.aisoftoj.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nan.aisoftoj.dto.recommendation.KnowledgeGraphDTO;
import com.nan.aisoftoj.dto.recommendation.KnowledgeGraphEdgeUpdateRequest;
import com.nan.aisoftoj.dto.recommendation.KnowledgeGraphNodeUpdateRequest;
import com.nan.aisoftoj.dto.recommendation.KnowledgePointRecommendationDTO;
import com.nan.aisoftoj.dto.recommendation.KnowledgePointSourceDTO;
import com.nan.aisoftoj.dto.recommendation.StudyRoadmapDTO;
import com.nan.aisoftoj.dto.recommendation.StudyRoadmapRequest;
import com.nan.aisoftoj.dto.recommendation.WrongQuestionEvidenceDTO;
import com.nan.aisoftoj.entity.KnowledgeBase;
import com.nan.aisoftoj.mapper.KnowledgeBaseMapper;
import com.nan.aisoftoj.mapper.UserWrongQuestionStatMapper;
import com.nan.aisoftoj.service.RecommendationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

@Service
public class RecommendationServiceImpl implements RecommendationService {
    private static final Logger log = LoggerFactory.getLogger(RecommendationServiceImpl.class);
    private final UserWrongQuestionStatMapper wrongQuestionStatMapper;
    private final KnowledgeBaseMapper knowledgeBaseMapper;
    private final Neo4jRecommendationGraphClient graphClient;
    private final ObjectMapper objectMapper;
    private final Map<String, Long> alignmentAttempts = new ConcurrentHashMap<>();
    private final Map<String, Integer> alignmentOffsets = new ConcurrentHashMap<>();

    @Value("${ai-service.url:http://localhost:8090}")
    private String aiServiceUrl;

    @Value("${ai-service.secret:}")
    private String aiServiceSecret;

    public RecommendationServiceImpl(
            UserWrongQuestionStatMapper wrongQuestionStatMapper,
            KnowledgeBaseMapper knowledgeBaseMapper,
            Neo4jRecommendationGraphClient graphClient,
            ObjectMapper objectMapper) {
        this.wrongQuestionStatMapper = wrongQuestionStatMapper;
        this.knowledgeBaseMapper = knowledgeBaseMapper;
        this.graphClient = graphClient;
        this.objectMapper = objectMapper;
    }

    @Override
    public List<KnowledgePointRecommendationDTO> listKnowledgePointRecommendations(Integer userId) {
        return listKnowledgePointRecommendations(userId, null);
    }

    @Override
    public List<KnowledgePointRecommendationDTO> listKnowledgePointRecommendations(
            Integer userId,
            Long knowledgeBaseId) {
        List<WrongQuestionEvidenceDTO> evidences = wrongQuestionStatMapper.selectRecommendationEvidence(userId);
        KnowledgeBase knowledgeBase = resolveKnowledgeBase(userId, knowledgeBaseId);
        if (knowledgeBase == null || evidences.isEmpty()) {
            return buildRecommendations(evidences, 8, knowledgeBase == null ? null : knowledgeBase.getId());
        }
        graphClient.replaceWrongQuestionEvidence(userId, evidences);
        ensureDocumentAlignments(userId, knowledgeBase, evidences);
        List<Map<String, Object>> alignments = graphClient.loadUserDocumentAlignments(
                userId,
                knowledgeBase.getVectorId());
        List<Map<String, Object>> relations = graphClient.loadUserDocumentRelations(
                userId,
                knowledgeBase.getVectorId());
        return buildDocumentRecommendations(evidences, alignments, relations, knowledgeBase, 8);
    }

    @Override
    public KnowledgeGraphDTO getKnowledgeGraph(Integer userId, String scope) {
        return getKnowledgeGraph(userId, scope, null);
    }

    @Override
    public KnowledgeGraphDTO getKnowledgeGraph(
            Integer userId,
            String scope,
            Long knowledgeBaseId) {
        List<WrongQuestionEvidenceDTO> evidences = wrongQuestionStatMapper.selectRecommendationEvidence(userId);
        KnowledgeBase knowledgeBase = resolveKnowledgeBase(userId, knowledgeBaseId);
        String vectorId = knowledgeBase == null ? null : knowledgeBase.getVectorId();
        graphClient.replaceWrongQuestionEvidence(userId, evidences);
        if (knowledgeBase == null || vectorId == null || vectorId.trim().isEmpty()) {
            KnowledgeGraphDTO graph = new KnowledgeGraphDTO();
            graph.setGraphAvailable(false);
            graph.setSource("no_knowledge_base_selected");
            return graph;
        }
        if ("full".equalsIgnoreCase(scope)) {
            return graphClient.loadUserFullGraph(userId, vectorId);
        }
        if (evidences.isEmpty()) {
            KnowledgeGraphDTO graph = new KnowledgeGraphDTO();
            graph.setGraphAvailable(false);
            graph.setSource("no_wrong_question_evidence");
            return graph;
        }
        KnowledgeGraphDTO graph = graphClient.loadUserWeakGraph(userId, vectorId);
        if (graph.getNodes().isEmpty()) {
            graph.setGraphAvailable(false);
            graph.setSource("no_document_knowledge_alignment");
        }
        return graph;
    }

    @Override
    public KnowledgeGraphDTO updateKnowledgeGraphNode(
            Integer userId,
            String nodeId,
            KnowledgeGraphNodeUpdateRequest request) {
        graphClient.updateKnowledgeNode(userId, nodeId, request.getLabel());
        return graphClient.loadUserWeakGraph(userId);
    }

    @Override
    public KnowledgeGraphDTO updateKnowledgeGraphEdge(
            Integer userId,
            String edgeId,
            KnowledgeGraphEdgeUpdateRequest request) {
        graphClient.updateKnowledgeEdge(
                userId,
                edgeId,
                request.getType(),
                request.getLabel(),
                request.getWeight());
        return graphClient.loadUserWeakGraph(userId);
    }

    @Override
    public KnowledgeGraphDTO deleteKnowledgeGraphEdge(Integer userId, String edgeId) {
        graphClient.deleteKnowledgeEdge(userId, edgeId);
        return graphClient.loadUserWeakGraph(userId);
    }

    @Override
    public StudyRoadmapDTO generateStudyRoadmap(Integer userId, StudyRoadmapRequest request) {
        int days = request == null || request.getDays() == null ? 7 : request.getDays();
        if (days != 14) {
            days = 7;
        }
        int dailyMinutes = request == null || request.getDailyMinutes() == null
                ? 60
                : Math.max(20, Math.min(request.getDailyMinutes(), 180));
        List<KnowledgePointRecommendationDTO> recommendations =
                listKnowledgePointRecommendations(userId, null);
        return requestRoadmapAgent(days, dailyMinutes, recommendations);
    }

    private KnowledgeBase resolveKnowledgeBase(Integer userId, Long requestedId) {
        LambdaQueryWrapper<KnowledgeBase> query = new LambdaQueryWrapper<KnowledgeBase>()
                .eq(KnowledgeBase::getUserId, Long.valueOf(userId));
        if (requestedId != null) {
            query.eq(KnowledgeBase::getId, requestedId);
            KnowledgeBase selected = knowledgeBaseMapper.selectOne(query);
            if (selected == null) {
                throw new IllegalArgumentException("知识库不存在或不属于当前用户");
            }
            return selected;
        }
        query.orderByDesc(KnowledgeBase::getIsDefault)
                .orderByAsc(KnowledgeBase::getCreateTime)
                .last("LIMIT 1");
        return knowledgeBaseMapper.selectOne(query);
    }

    private void ensureDocumentAlignments(
            Integer userId,
            KnowledgeBase knowledgeBase,
            List<WrongQuestionEvidenceDTO> evidences) {
        String vectorId = knowledgeBase.getVectorId();
        if (vectorId == null || vectorId.trim().isEmpty()) {
            return;
        }
        List<Map<String, Object>> existing = graphClient.loadUserDocumentAlignments(userId, vectorId);
        Set<String> mappedQuestionIds = new HashSet<>();
        for (Map<String, Object> alignment : existing) {
            mappedQuestionIds.add(string(alignment.get("questionId")));
        }
        List<WrongQuestionEvidenceDTO> allUnmapped = new ArrayList<>();
        for (WrongQuestionEvidenceDTO evidence : evidences) {
            String questionId = evidence.getQuestionId() == null
                    ? ""
                    : String.valueOf(evidence.getQuestionId());
            if (!mappedQuestionIds.contains(questionId)) {
                allUnmapped.add(evidence);
            }
        }
        if (allUnmapped.isEmpty()) {
            return;
        }

        String attemptKey = userId + ":" + knowledgeBase.getId();
        long now = System.currentTimeMillis();
        AtomicBoolean permitted = new AtomicBoolean(false);
        alignmentAttempts.compute(attemptKey, (key, previousAttempt) -> {
            if (previousAttempt == null || now - previousAttempt >= 60_000L) {
                permitted.set(true);
                return now;
            }
            return previousAttempt;
        });
        if (!permitted.get()) {
            return;
        }

        int start = Math.floorMod(alignmentOffsets.getOrDefault(attemptKey, 0), allUnmapped.size());
        int windowSize = Math.min(40, allUnmapped.size());
        List<WrongQuestionEvidenceDTO> unmapped = new ArrayList<>(windowSize);
        for (int index = 0; index < windowSize; index++) {
            unmapped.add(allUnmapped.get((start + index) % allUnmapped.size()));
        }
        alignmentOffsets.put(attemptKey, (start + windowSize) % allUnmapped.size());

        Map<String, Object> graphPayload = graphClient.loadUserKnowledgeBaseAlignmentPayload(
                userId,
                vectorId);
        if (asMapList(graphPayload.get("entity_nodes")).isEmpty()) {
            return;
        }
        try {
            Map<String, Object> body = new HashMap<>();
            body.put("user_id", String.valueOf(userId));
            body.put("document_id", "knowledge-base-" + knowledgeBase.getId());
            body.put("wrong_questions", unmapped);
            body.put("entity_nodes", graphPayload.get("entity_nodes"));
            body.put("kg_extraction_chunks", graphPayload.get("kg_extraction_chunks"));
            body.put("max_alignments", 120);

            URL endpoint = new URL(aiServiceUrl.replaceAll("/+$", "")
                    + "/api/v1/knowledge-graph/wrong-question-alignments");
            HttpURLConnection connection = (HttpURLConnection) endpoint.openConnection();
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(10_000);
            connection.setReadTimeout(120_000);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            if (aiServiceSecret != null && !aiServiceSecret.isEmpty()) {
                connection.setRequestProperty("X-Aisoftoj-Internal-Secret", aiServiceSecret);
            }
            try (OutputStream output = connection.getOutputStream()) {
                output.write(objectMapper.writeValueAsBytes(body));
            }
            int responseCode = connection.getResponseCode();
            if (responseCode < 200 || responseCode >= 300) {
                log.warn("Qwen alignment request failed with HTTP {} for user {} knowledge base {}",
                        responseCode, userId, knowledgeBase.getId());
                return;
            }
            Map<String, Object> response;
            try (InputStream input = connection.getInputStream()) {
                response = objectMapper.readValue(
                        input,
                        new com.fasterxml.jackson.core.type.TypeReference<Map<String, Object>>() {});
            }
            graphClient.syncUserKnowledgeBaseAlignments(
                    userId,
                    vectorId,
                    asMapList(response.get("alignments")));
        } catch (Exception exception) {
            log.warn("Qwen alignment request failed for user {} knowledge base {}",
                    userId, knowledgeBase.getId(), exception);
        }
    }

    private List<KnowledgePointRecommendationDTO> buildDocumentRecommendations(
            List<WrongQuestionEvidenceDTO> evidences,
            List<Map<String, Object>> alignments,
            List<Map<String, Object>> relations,
            KnowledgeBase knowledgeBase,
            int limit) {
        Map<String, WrongQuestionEvidenceDTO> evidenceByQuestion = new HashMap<>();
        for (WrongQuestionEvidenceDTO evidence : evidences) {
            if (evidence.getQuestionId() != null) {
                evidenceByQuestion.put(String.valueOf(evidence.getQuestionId()), evidence);
            }
        }

        Map<String, KnowledgePointRecommendationDTO> grouped = new LinkedHashMap<>();
        Map<String, String> groupByEntityId = new HashMap<>();
        Map<String, Set<String>> sourceKeysByGroup = new HashMap<>();
        Set<String> mappedQuestionIds = new HashSet<>();
        Set<String> countedQuestionGroups = new HashSet<>();

        for (Map<String, Object> alignment : alignments) {
            String questionId = string(alignment.get("questionId"));
            WrongQuestionEvidenceDTO evidence = evidenceByQuestion.get(questionId);
            String name = string(alignment.get("name"));
            String entityId = string(alignment.get("knowledgePointId"));
            if (evidence == null || name.isEmpty() || entityId.isEmpty()) {
                continue;
            }
            // Document-scoped entity IDs include version and structural scope.
            // Grouping by display name would merge homonyms that extraction
            // deliberately kept separate.
            String groupKey = entityId;
            KnowledgePointRecommendationDTO item = grouped.get(groupKey);
            if (item == null) {
                item = new KnowledgePointRecommendationDTO();
                item.setId(entityId);
                item.setName(name);
                item.setSubject(firstNonBlank(
                        evidence.getSubjectName(),
                        string(alignment.get("subject")),
                        "通用"));
                item.setCategory(name);
                item.setScore(0);
                item.setErrorCount(0);
                item.setWrongQuestionCount(0);
                item.setSourceType("document_knowledge_graph");
                item.setKnowledgeBaseId(knowledgeBase.getId());
                grouped.put(groupKey, item);
            }

            groupByEntityId.put(entityId, groupKey);
            mappedQuestionIds.add(questionId);
            String countedKey = groupKey + "|" + questionId;
            if (countedQuestionGroups.add(countedKey)) {
                int errorCount = evidence.getErrorCount() == null ? 1 : evidence.getErrorCount();
                double confidence = number(alignment.get("confidence"), 0.6);
                item.setErrorCount(item.getErrorCount() + errorCount);
                item.setWrongQuestionCount(item.getWrongQuestionCount() + 1);
                item.setScore(item.getScore() + (int) Math.round(scoreEvidence(evidence) * confidence));
                item.getEvidences().add(evidence);
            }

            KnowledgePointSourceDTO source = buildSource(alignment);
            String sourceKey = source.getDocumentId() + "|" + source.getSourcePageRange()
                    + "|" + String.join(">", source.getHeadingPath());
            if (sourceKeysByGroup.computeIfAbsent(groupKey, key -> new HashSet<>()).add(sourceKey)) {
                item.getSources().add(source);
            }
        }

        applyDocumentRelations(grouped, groupByEntityId, relations);
        List<KnowledgePointRecommendationDTO> result = new ArrayList<>(grouped.values());
        for (KnowledgePointRecommendationDTO item : result) {
            int score = Math.min(100, 25 + item.getScore());
            item.setScore(score);
            item.setMastery(estimateMastery(score, item.getWrongQuestionCount()));
            item.setLevel(resolveLevel(score));
            item.setReason("根据知识库原文映射到 " + item.getSources().size()
                    + " 处文档证据，命中 " + item.getWrongQuestionCount() + " 道错题。");
            item.setSuggestion("先回到原文对应章节复习，再重做关联错题并检查易混淆知识点。");
        }

        List<WrongQuestionEvidenceDTO> unmapped = new ArrayList<>();
        for (WrongQuestionEvidenceDTO evidence : evidences) {
            String questionId = evidence.getQuestionId() == null
                    ? ""
                    : String.valueOf(evidence.getQuestionId());
            if (!mappedQuestionIds.contains(questionId)) {
                unmapped.add(evidence);
            }
        }
        result.addAll(buildRecommendations(unmapped, limit, knowledgeBase.getId()));
        result.sort(Comparator.comparing(KnowledgePointRecommendationDTO::getScore).reversed());
        if (result.size() > limit) {
            return new ArrayList<>(result.subList(0, limit));
        }
        return result;
    }

    private KnowledgePointSourceDTO buildSource(Map<String, Object> alignment) {
        KnowledgePointSourceDTO source = new KnowledgePointSourceDTO();
        source.setDocumentId(string(alignment.get("documentId")));
        source.setDocumentName(string(alignment.get("documentName")));
        source.setSourcePageRange(string(alignment.get("sourcePageRange")));
        source.setHeadingPath(stringList(alignment.get("headingPath")));
        source.setConfidence(number(alignment.get("confidence"), 0.0));
        source.setEvidence(firstNonBlank(
                string(alignment.get("evidence")),
                string(alignment.get("reason"))));
        return source;
    }

    private void applyDocumentRelations(
            Map<String, KnowledgePointRecommendationDTO> grouped,
            Map<String, String> groupByEntityId,
            List<Map<String, Object>> relations) {
        for (Map<String, Object> relation : relations) {
            String sourceId = string(relation.get("sourceId"));
            String targetId = string(relation.get("targetId"));
            String sourceGroup = groupByEntityId.get(sourceId);
            String targetGroup = groupByEntityId.get(targetId);
            String type = string(relation.get("relationType"));
            if ("PREREQUISITE_OF".equals(type) && targetGroup != null) {
                addUnique(grouped.get(targetGroup).getPrerequisiteNames(), string(relation.get("sourceName")));
            }
            if (sourceGroup != null) {
                addUnique(grouped.get(sourceGroup).getRelatedNames(), string(relation.get("targetName")));
            }
            if (targetGroup != null && !"PREREQUISITE_OF".equals(type)) {
                addUnique(grouped.get(targetGroup).getRelatedNames(), string(relation.get("sourceName")));
            }
        }
    }

    private void addUnique(List<String> values, String value) {
        if (value != null && !value.trim().isEmpty() && !values.contains(value.trim())) {
            values.add(value.trim());
        }
    }

    private List<KnowledgePointRecommendationDTO> buildRecommendations(
            List<WrongQuestionEvidenceDTO> evidences,
            int limit) {
        return buildRecommendations(evidences, limit, null);
    }

    private List<KnowledgePointRecommendationDTO> buildRecommendations(
            List<WrongQuestionEvidenceDTO> evidences,
            int limit,
            Long knowledgeBaseId) {
        Map<String, KnowledgePointRecommendationDTO> grouped = new LinkedHashMap<>();
        for (WrongQuestionEvidenceDTO evidence : evidences) {
            String knowledgeName = firstNonBlank(evidence.getKnowledgePointName(), evidence.getQuestionName(), "未归类知识点");
            String id = "legacy:kp:" + firstNonBlank(evidence.getSubjectName(), "通用") + ":" + knowledgeName;
            KnowledgePointRecommendationDTO item = grouped.get(id);
            if (item == null) {
                item = new KnowledgePointRecommendationDTO();
                item.setId(id);
                item.setName(knowledgeName);
                item.setSubject(firstNonBlank(evidence.getSubjectName(), "通用"));
                item.setCategory(knowledgeName);
                item.setScore(0);
                item.setErrorCount(0);
                item.setWrongQuestionCount(0);
                item.setSourceType("category_fallback");
                item.setKnowledgeBaseId(knowledgeBaseId);
                item.setEvidences(new ArrayList<>());
                grouped.put(id, item);
            }
            int errorCount = evidence.getErrorCount() == null ? 1 : evidence.getErrorCount();
            item.setErrorCount(item.getErrorCount() + errorCount);
            item.setWrongQuestionCount(item.getWrongQuestionCount() + 1);
            item.setScore(item.getScore() + scoreEvidence(evidence));
            item.getEvidences().add(evidence);
        }
        List<KnowledgePointRecommendationDTO> result = new ArrayList<>(grouped.values());
        result.sort(Comparator.comparing(KnowledgePointRecommendationDTO::getScore).reversed());
        if (result.size() > limit) {
            result = new ArrayList<>(result.subList(0, limit));
        }
        for (KnowledgePointRecommendationDTO item : result) {
            int normalizedScore = normalizeRecommendationScore(item.getScore());
            item.setScore(normalizedScore);
            item.setMastery(estimateMastery(normalizedScore, item.getWrongQuestionCount()));
            item.setLevel(resolveLevel(item.getScore()));
            item.setReason("当前没有可靠的文档知识点映射，暂按题库分类聚合：命中 "
                    + item.getWrongQuestionCount() + " 道错题，累计错误 "
                    + item.getErrorCount() + " 次。");
            item.setSuggestion("先回看关联错题，再按前置概念、核心概念、同类题三步复习。");
        }
        return result;
    }

    private StudyRoadmapDTO requestRoadmapAgent(
            int days,
            int dailyMinutes,
            List<KnowledgePointRecommendationDTO> recommendations) {
        try {
            URL endpoint = new URL(aiServiceUrl.replaceAll("/+$", "")
                    + "/api/v1/recommendations/study-roadmap");
            HttpURLConnection connection = (HttpURLConnection) endpoint.openConnection();
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(10_000);
            connection.setReadTimeout(60_000);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            if (aiServiceSecret != null && !aiServiceSecret.isEmpty()) {
                connection.setRequestProperty("X-Aisoftoj-Internal-Secret", aiServiceSecret);
            }
            Map<String, Object> body = new HashMap<>();
            body.put("days", days);
            body.put("daily_minutes", dailyMinutes);
            body.put("recommendations", recommendations);
            byte[] payload = objectMapper.writeValueAsBytes(body);
            try (OutputStream output = connection.getOutputStream()) {
                output.write(payload);
            }
            if (connection.getResponseCode() < 200 || connection.getResponseCode() >= 300) {
                throw new IllegalStateException("AI service returned " + connection.getResponseCode());
            }
            try (InputStream input = connection.getInputStream()) {
                return objectMapper.readValue(input, StudyRoadmapDTO.class);
            }
        } catch (Exception exception) {
            throw new IllegalStateException("学习路线 Agent 暂时不可用: " + exception.getMessage(), exception);
        }
    }

    private int scoreEvidence(WrongQuestionEvidenceDTO evidence) {
        int errorCount = Math.max(1, evidence.getErrorCount() == null ? 1 : evidence.getErrorCount());
        int difficulty = Math.max(1, evidence.getDifficulty() == null ? 2 : evidence.getDifficulty());
        int score = 10 + (int) Math.round(Math.log1p(errorCount) * 12 + Math.sqrt(errorCount) * 4);
        score += difficulty * 4;
        String importance = evidence.getImportanceLevel();
        if ("must".equals(importance)) score += 14;
        else if ("high".equals(importance)) score += 10;
        else if ("medium".equals(importance)) score += 6;
        return score;
    }

    private int normalizeRecommendationScore(int rawScore) {
        if (rawScore <= 0) {
            return 0;
        }
        return Math.min(100, (int) Math.round(25 + Math.log1p(rawScore) * 13));
    }

    private int estimateMastery(int score, int wrongQuestionCount) {
        int penalty = Math.min(12, Math.max(0, wrongQuestionCount));
        int mastery = (int) Math.round(92 - score * 0.68 - penalty);
        return Math.max(12, Math.min(88, mastery));
    }

    private String resolveLevel(int score) {
        if (score >= 85) return "must";
        if (score >= 68) return "high";
        if (score >= 48) return "medium";
        return "low";
    }

    private String firstNonBlank(String first, String second, String defaultValue) {
        if (first != null && !first.trim().isEmpty()) return first.trim();
        if (second != null && !second.trim().isEmpty()) return second.trim();
        return defaultValue;
    }

    private String firstNonBlank(String first, String defaultValue) {
        if (first != null && !first.trim().isEmpty()) return first.trim();
        return defaultValue;
    }

    private String string(Object value) {
        return value == null ? "" : String.valueOf(value).trim();
    }

    private double number(Object value, double defaultValue) {
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        try {
            return Double.parseDouble(string(value));
        } catch (NumberFormatException exception) {
            return defaultValue;
        }
    }

    private List<String> stringList(Object value) {
        List<String> result = new ArrayList<>();
        if (value instanceof List) {
            for (Object item : (List<?>) value) {
                String text = string(item);
                if (!text.isEmpty()) {
                    result.add(text);
                }
            }
        }
        return result;
    }

    private List<Map<String, Object>> asMapList(Object value) {
        List<Map<String, Object>> result = new ArrayList<>();
        if (!(value instanceof List)) {
            return result;
        }
        for (Object item : (List<?>) value) {
            if (!(item instanceof Map)) {
                continue;
            }
            Map<String, Object> converted = new HashMap<>();
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) item).entrySet()) {
                converted.put(String.valueOf(entry.getKey()), entry.getValue());
            }
            result.add(converted);
        }
        return result;
    }
}
