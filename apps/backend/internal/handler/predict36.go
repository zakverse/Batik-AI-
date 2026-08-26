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

type Predict36Handler struct {
	predictService  *service.Predict36Service
	defaultTopK     int
	maxUploadSizeMB int64
}

func NewPredict36Handler(predictService *service.Predict36Service, defaultTopK int, maxUploadSizeMB int64) *Predict36Handler {
	return &Predict36Handler{
		predictService:  predictService,
		defaultTopK:     defaultTopK,
		maxUploadSizeMB: maxUploadSizeMB,
	}
}

func (h *Predict36Handler) Predict36(c *gin.Context) {
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

	// 4. Validate top_k parameter (1..36)
	topK := h.defaultTopK
	topKParam := strings.TrimSpace(c.DefaultQuery("top_k", c.PostForm("top_k")))
	if topKParam != "" {
		parsedK, err := strconv.Atoi(topKParam)
		if err != nil || parsedK < 1 || parsedK > 36 {
			c.JSON(http.StatusBadRequest, inference.ErrorResponse{
				Success: false,
				Error: inference.APIErrorDetail{
					Code:    "INVALID_TOP_K",
					Message: "Parameter 'top_k' must be an integer between 1 and 36",
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
	result, err := h.predictService.PredictImage36(file, topK)
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
	fmt.Printf("[INFERENCE 36-CLASS] %s -> Class: %s (ID: %d, Conf: %.4f, IsBatik: %t) in %.2f ms\n",
		fileHeader.Filename,
		result.Prediction.Label,
		result.Prediction.ClassID,
		result.Prediction.Confidence,
		result.Prediction.IsBatik,
		float64(duration.Microseconds())/1000.0,
	)

	c.JSON(http.StatusOK, result)
}
