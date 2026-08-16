package handler

import (
	"fmt"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"wastra-ai/backend/internal/inference"
	"wastra-ai/backend/internal/service"

	"github.com/gin-gonic/gin"
)

type PredictHandler struct {
	predictService  *service.PredictService
	defaultTopK     int
	maxUploadSizeMB int64
}

func NewPredictHandler(predictService *service.PredictService, defaultTopK int, maxUploadSizeMB int64) *PredictHandler {
	return &PredictHandler{
		predictService:  predictService,
		defaultTopK:     defaultTopK,
		maxUploadSizeMB: maxUploadSizeMB,
	}
}

func (h *PredictHandler) Predict(c *gin.Context) {
	// 1. Extract File from multipart form field "image"
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

	// 2. Validate File Size (> 0 and <= MaxUploadSizeMB)
	if fileHeader.Size == 0 {
		c.JSON(http.StatusBadRequest, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "EMPTY_IMAGE",
				Message: "Uploaded image file is empty (0 bytes)",
			},
		})
		return
	}

	maxBytes := h.maxUploadSizeMB << 20
	if fileHeader.Size > maxBytes {
		c.JSON(http.StatusRequestEntityTooLarge, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "FILE_TOO_LARGE",
				Message: fmt.Sprintf("Image size (%d bytes) exceeds the maximum allowed limit of %d MB", fileHeader.Size, h.maxUploadSizeMB),
			},
		})
		return
	}

	// 3. Validate File Extension
	ext := strings.ToLower(filepath.Ext(fileHeader.Filename))
	validExts := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".webp": true,
	}
	if !validExts[ext] {
		c.JSON(http.StatusBadRequest, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "UNSUPPORTED_FORMAT",
				Message: fmt.Sprintf("Unsupported file extension '%s'. Only JPEG, PNG, and WebP images are allowed.", ext),
			},
		})
		return
	}

	// 4. Validate top_k parameter
	topK := h.defaultTopK
	topKParam := strings.TrimSpace(c.DefaultQuery("top_k", c.PostForm("top_k")))
	if topKParam != "" {
		parsedK, err := strconv.Atoi(topKParam)
		if err != nil || parsedK < 1 || parsedK > 35 {
			c.JSON(http.StatusBadRequest, inference.ErrorResponse{
				Success: false,
				Error: inference.APIErrorDetail{
					Code:    "INVALID_TOP_K",
					Message: "Parameter 'top_k' must be an integer between 1 and 35",
				},
			})
			return
		}
		topK = parsedK
	}

	// 5. Open File Stream
	file, err := fileHeader.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "INTERNAL_ERROR",
				Message: "Failed to open uploaded image for reading",
			},
		})
		return
	}
	defer file.Close()

	// 6. Execute Prediction Service
	t0 := time.Now()
	result, err := h.predictService.PredictImage(file, topK)
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
	duration := time.Since(t0)

	// Logging prediction summary
	fmt.Printf("[INFERENCE] %s -> Class: %s (Confidence: %.4f) in %.2f ms\n",
		fileHeader.Filename,
		result.Prediction.Class,
		result.Prediction.Confidence,
		float64(duration.Microseconds())/1000.0,
	)

	c.JSON(http.StatusOK, result)
}
