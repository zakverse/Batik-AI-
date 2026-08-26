package handler

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"net/http"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"wastra-ai/backend/internal/inference"
	"wastra-ai/backend/internal/service"

	"github.com/gin-gonic/gin"
)

type Benchmark36Handler struct {
	predictService *service.Predict36Service
}

func NewBenchmark36Handler(predictService *service.Predict36Service) *Benchmark36Handler {
	return &Benchmark36Handler{
		predictService: predictService,
	}
}

func (h *Benchmark36Handler) Benchmark36(c *gin.Context) {
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

	ext := strings.ToLower(filepath.Ext(fileHeader.Filename))
	if ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".webp" {
		c.JSON(http.StatusBadRequest, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "UNSUPPORTED_FORMAT",
				Message: "Only JPEG, PNG, and WebP images are supported for benchmarking.",
			},
		})
		return
	}

	iterations := 20
	iterParam := strings.TrimSpace(c.DefaultQuery("iterations", c.PostForm("iterations")))
	if iterParam != "" {
		parsed, err := strconv.Atoi(iterParam)
		if err == nil && parsed >= 1 && parsed <= 200 {
			iterations = parsed
		}
	}

	file, err := fileHeader.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "FILE_OPEN_ERROR",
				Message: "Failed to open uploaded image",
			},
		})
		return
	}
	defer file.Close()

	imgBytes, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "FILE_READ_ERROR",
				Message: "Failed to read image bytes",
			},
		})
		return
	}

	// 1. Preprocess once
	tensor, err := inference.PreprocessImage(bytes.NewReader(imgBytes))
	if err != nil {
		c.JSON(http.StatusBadRequest, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "PREPROCESS_FAILED",
				Message: err.Error(),
			},
		})
		return
	}

	engine := h.predictService.GetEngine()

	// Warm-up 3 runs
	for i := 0; i < 3; i++ {
		_, _, _ = engine.Predict36(tensor, 1)
	}

	latencies := make([]float64, iterations)
	var lastPred *inference.Prediction36Item

	tTotalStart := time.Now()
	for i := 0; i < iterations; i++ {
		t0 := time.Now()
		pred, _, err := engine.Predict36(tensor, 1)
		if err != nil {
			c.JSON(http.StatusInternalServerError, inference.ErrorResponse{
				Success: false,
				Error: inference.APIErrorDetail{
					Code:    "INFERENCE_FAILED",
					Message: fmt.Sprintf("Benchmark iteration %d failed: %v", i+1, err),
				},
			})
			return
		}
		latencies[i] = float64(time.Since(t0).Microseconds()) / 1000.0
		lastPred = pred
	}
	totalTimeMs := float64(time.Since(tTotalStart).Microseconds()) / 1000.0

	sumLat := 0.0
	for _, l := range latencies {
		sumLat += l
	}
	avgLat := sumLat / float64(iterations)

	sort.Float64s(latencies)
	p50Lat := latencies[iterations/2]
	fps := (float64(iterations) / totalTimeMs) * 1000.0

	c.JSON(http.StatusOK, inference.Benchmark36Response{
		Success: true,
		Model: inference.Model36Info{
			Name:    "efficientnetb0",
			Version: "36-class",
			Runtime: "onnx",
		},
		SamplesCount:  iterations,
		TotalTimeMs:   math.Round(totalTimeMs*100) / 100,
		AvgLatencyMs:  math.Round(avgLat*100) / 100,
		P50LatencyMs:  math.Round(p50Lat*100) / 100,
		ThroughputFPS: math.Round(fps*100) / 100,
		SamplePred:    *lastPred,
	})
}
