package handler

import (
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"wastra-ai/backend/internal/inference"
	"wastra-ai/backend/internal/service"

	"github.com/gin-gonic/gin"
)

type PredictHandler struct {
	predictService *service.PredictService
	defaultTopK    int
}

func NewPredictHandler(predictService *service.PredictService, defaultTopK int) *PredictHandler {
	return &PredictHandler{
		predictService: predictService,
		defaultTopK:    defaultTopK,
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

	// 2. Validate File Size (> 0)
	if fileHeader.Size == 0 {
		c.JSON(http.StatusBadRequest, inference.ErrorResponse{
			Success: false,
			Error: inference.APIErrorDetail{
				Code:    "EMPTY_FILE",
				Message: "Uploaded image file is empty (0 bytes)",
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
				Message: "Unsupported file extension. Only JPEG, PNG, and WebP are allowed.",
			},
		})
		return
	}

	// 4. Open File Stream
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

	// 5. Parse optional top_k parameter
	topK := h.defaultTopK
	if topKStr := c.DefaultQuery("top_k", c.PostForm("top_k")); topKStr != "" {
		if parsedK, err := strconv.Atoi(topKStr); err == nil && parsedK > 0 {
			topK = parsedK
		}
	}

	// 6. Execute Prediction Service
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

	c.JSON(http.StatusOK, result)
}
