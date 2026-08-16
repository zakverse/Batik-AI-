package handler

import (
	"fmt"
	"net/http"
	"wastra-ai/backend/internal/inference"

	"github.com/gin-gonic/gin"
)

type HealthHandler struct {
	engine *inference.Engine
}

func NewHealthHandler(engine *inference.Engine) *HealthHandler {
	return &HealthHandler{
		engine: engine,
	}
}

func (h *HealthHandler) HealthCheck(c *gin.Context) {
	meta := h.engine.GetMetadata()
	c.JSON(http.StatusOK, inference.HealthResponse{
		Status:     "ok",
		Model:      meta.ModelName,
		Classes:    h.engine.GetClassCount(),
		Version:    "1.0.0",
		Accuracy:   fmt.Sprintf("%.2f%%", meta.TestAccuracy*100),
		Generalize: "-0.58 pp (Pass)",
	})
}
