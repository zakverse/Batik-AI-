package handler

import (
	"fmt"
	"net/http"
	"wastra-ai/backend/internal/inference"

	"github.com/gin-gonic/gin"
)

type Health36Handler struct {
	engine *inference.Engine36
}

func NewHealth36Handler(engine *inference.Engine36) *Health36Handler {
	return &Health36Handler{
		engine: engine,
	}
}

func (h *Health36Handler) HealthCheck(c *gin.Context) {
	meta := h.engine.GetMetadata()
	acc := "86.48%"
	macroF1 := 0.8707
	weightedF1 := 0.8637
	nonBatikF1 := 1.0000

	if testMetrics, ok := meta["test_evaluation_metrics"].(map[string]interface{}); ok {
		if accVal, ok := testMetrics["accuracy_percent"].(float64); ok {
			acc = fmt.Sprintf("%.2f%%", accVal)
		}
		if mf1, ok := testMetrics["macro_f1"].(float64); ok {
			macroF1 = mf1
		}
		if wf1, ok := testMetrics["weighted_f1"].(float64); ok {
			weightedF1 = wf1
		}
		if nbPerf, ok := testMetrics["non_batik_performance"].(map[string]interface{}); ok {
			if nbf1, ok := nbPerf["f1_score"].(float64); ok {
				nonBatikF1 = nbf1
			}
		}
	}

	c.JSON(http.StatusOK, inference.Health36Response{
		Status:     "ok",
		Model:      "EfficientNetB0 36-Class Fine-Tuned",
		Version:    "36-class (35 Batik + 1 Non-Batik)",
		Classes:    h.engine.GetClassCount(),
		Accuracy:   acc,
		MacroF1:    macroF1,
		WeightedF1: weightedF1,
		NonBatikF1: nonBatikF1,
	})
}
