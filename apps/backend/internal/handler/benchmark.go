package handler

import (
	"bytes"
	"io"
	"net/http"
	"strconv"
	"time"
	"wastra-ai/backend/internal/inference"
	"wastra-ai/backend/internal/service"

	"github.com/gin-gonic/gin"
)

type BenchmarkHandler struct {
	predictService *service.PredictService
}

func NewBenchmarkHandler(predictService *service.PredictService) *BenchmarkHandler {
	return &BenchmarkHandler{
		predictService: predictService,
	}
}

func (h *BenchmarkHandler) Benchmark(c *gin.Context) {
	fileHeader, err := c.FormFile("image")
	if err != nil {
		c.JSON(http.StatusBadRequest, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "NO_IMAGE_UPLOADED",
				Message: "Image file is required under multipart form field 'image'",
			},
		})
		return
	}

	iterations := 50
	if itersStr := c.DefaultQuery("iterations", c.PostForm("iterations")); itersStr != "" {
		if parsedIters, err := strconv.Atoi(itersStr); err == nil && parsedIters > 0 {
			iterations = parsedIters
		}
	}

	file, err := fileHeader.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "FILE_READ_ERROR",
				Message: "Failed to open uploaded image",
			},
		})
		return
	}
	defer file.Close()

	imgBytes, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusBadRequest, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "FILE_READ_ERROR",
				Message: "Failed to read image bytes",
			},
		})
		return
	}

	// Preprocess once
	tensor, err := inference.PreprocessImage(bytes.NewReader(imgBytes))
	if err != nil {
		c.JSON(http.StatusBadRequest, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "INVALID_IMAGE",
				Message: err.Error(),
			},
		})
		return
	}

	engine := h.predictService.GetEngine()

	// Warmup
	_, _, _ = engine.Predict(tensor, 1)

	// Run benchmark loop
	start := time.Now()
	var lastPred *inference.PredictionItem
	for i := 0; i < iterations; i++ {
		top1, _, err := engine.Predict(tensor, 3)
		if err != nil {
			c.JSON(http.StatusInternalServerError, inference.ErrorResponse{
				Success: false,
				Error: inference.APIErrorDetail{
					Code:    "INFERENCE_ERROR",
					Message: err.Error(),
				},
			})
			return
		}
		lastPred = top1
	}
	totalDuration := time.Since(start)

	totalMs := float64(totalDuration.Microseconds()) / 1000.0
	avgLatencyMs := totalMs / float64(iterations)
	throughput := float64(iterations) / totalDuration.Seconds()

	c.JSON(http.StatusOK, inference.BenchmarkResponse{
		Success:        true,
		SamplesCount:   iterations,
		TotalTimeMs:    totalMs,
		AvgLatencyMs:   avgLatencyMs,
		ThroughputFPS:  throughput,
		PredictedClass: lastPred.Class,
		Confidence:     lastPred.Confidence,
	})
}
