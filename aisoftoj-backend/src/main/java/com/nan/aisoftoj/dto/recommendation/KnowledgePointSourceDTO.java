package com.nan.aisoftoj.dto.recommendation;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class KnowledgePointSourceDTO {
    private String documentId;
    private String documentName;
    private String sourcePageRange;
    private List<String> headingPath = new ArrayList<>();
    private Double confidence;
    private String evidence;
}
