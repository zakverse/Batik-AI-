package handler_test

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"wastra-ai/backend/internal/config"
	"wastra-ai/backend/internal/inference"
	"wastra-ai/backend/routes"

	"github.com/gin-gonic/gin"
)

func findModelDir() string {
	candidates := []string{
		"model",
		"../model",
		"../../model",
		"../../../apps/backend/model",
		"apps/backend/model",
	}
	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(c, "efficientnetb0_finetuned.onnx")); err == nil {
			return c
		}
	}
	return "model"
}

func setupTestRouter(t *testing.T) (*gin.Engine, *inference.Engine) {
	gin.SetMode(gin.TestMode)
	modelDir := findModelDir()

	cfg := &config.Config{
		Port:             "8080",
		Host:             "0.0.0.0",
		ModelPath:        filepath.Join(modelDir, "efficientnetb0_finetuned.onnx"),
		ClassMappingPath: filepath.Join(modelDir, "efficientnetb0_class_mapping.json"),
		MetadataPath:     filepath.Join(modelDir, "efficientnetb0_model_metadata.json"),
		ONNXLibPath:      filepath.Join(modelDir, "onnxruntime.dll"),
		MaxUploadSizeMB:  10,
		DefaultTopK:      3,
		AllowedOrigins:   "*",
	}

	engine, err := inference.NewEngine(cfg.ModelPath, cfg.ClassMappingPath, cfg.MetadataPath, cfg.ONNXLibPath)
	if err != nil {
		t.Fatalf("Failed to initialize test engine: %v", err)
	}

	router := routes.SetupRouter(cfg, engine)
	return router, engine
}

func createMultipartRequest(fieldName, filename string, fileContent []byte) (*http.Request, string, error) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if fieldName != "" && len(fileContent) > 0 {
		part, err := writer.CreateFormFile(fieldName, filename)
		if err != nil {
			return nil, "", err
		}
		if _, err := io.Copy(part, bytes.NewReader(fileContent)); err != nil {
			return nil, "", err
		}
	}
	writer.Close()

	req := httptest.NewRequest(http.MethodPost, "/api/v1/predict", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return req, writer.FormDataContentType(), nil
}

func generateTestJPEG() []byte {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{R: 180, G: 80, B: 40, A: 255})
		}
	}
	var buf bytes.Buffer
	_ = jpeg.Encode(&buf, img, nil)
	return buf.Bytes()
}

func TestHealthEndpoint(t *testing.T) {
	router, engine := setupTestRouter(t)
	defer engine.Close()

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", rec.Code)
	}

	var res inference.HealthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if res.Status != "ok" || res.Classes != 35 {
		t.Errorf("Unexpected health response: %+v", res)
	}
}

func TestPredict_NoImage(t *testing.T) {
	router, engine := setupTestRouter(t)
	defer engine.Close()

	req, _, _ := createMultipartRequest("", "", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for missing image, got %d", rec.Code)
	}

	var errRes inference.ErrorResponse
	json.Unmarshal(rec.Body.Bytes(), &errRes)
	if errRes.Error.Code != "NO_IMAGE_UPLOADED" {
		t.Errorf("Expected error code NO_IMAGE_UPLOADED, got %s", errRes.Error.Code)
	}
}

func TestPredict_UnsupportedFormat(t *testing.T) {
	router, engine := setupTestRouter(t)
	defer engine.Close()

	req, _, _ := createMultipartRequest("image", "document.txt", []byte("plain text file"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for unsupported format, got %d", rec.Code)
	}

	var errRes inference.ErrorResponse
	json.Unmarshal(rec.Body.Bytes(), &errRes)
	if errRes.Error.Code != "UNSUPPORTED_FORMAT" {
		t.Errorf("Expected error code UNSUPPORTED_FORMAT, got %s", errRes.Error.Code)
	}
}

func TestPredict_InvalidTopK(t *testing.T) {
	router, engine := setupTestRouter(t)
	defer engine.Close()

	validJPEG := generateTestJPEG()
	req, _, _ := createMultipartRequest("image", "batik.jpg", validJPEG)
	req.URL.RawQuery = "top_k=50" // Exceeds max 35

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for top_k > 35, got %d", rec.Code)
	}

	var errRes inference.ErrorResponse
	json.Unmarshal(rec.Body.Bytes(), &errRes)
	if errRes.Error.Code != "INVALID_TOP_K" {
		t.Errorf("Expected error code INVALID_TOP_K, got %s", errRes.Error.Code)
	}
}

func TestPredict_ValidImage_DefaultTopK(t *testing.T) {
	router, engine := setupTestRouter(t)
	defer engine.Close()

	validJPEG := generateTestJPEG()
	req, _, _ := createMultipartRequest("image", "batik_sample.jpg", validJPEG)

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200 for valid image, got %d: %s", rec.Code, rec.Body.String())
	}

	var res inference.PredictSuccessResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatalf("Failed to decode success response: %v", err)
	}

	if !res.Success {
		t.Errorf("Expected success=true, got %v", res.Success)
	}

	if res.Prediction.Class == "" || res.Prediction.Confidence < 0.0 || res.Prediction.Confidence > 1.0 {
		t.Errorf("Invalid top1 prediction: %+v", res.Prediction)
	}

	if len(res.TopPredictions) != 3 {
		t.Errorf("Expected 3 top predictions by default, got %d", len(res.TopPredictions))
	}
}

func TestPredict_CustomTopK(t *testing.T) {
	router, engine := setupTestRouter(t)
	defer engine.Close()

	validJPEG := generateTestJPEG()
	req, _, _ := createMultipartRequest("image", "batik_sample.jpg", validJPEG)
	req.URL.RawQuery = "top_k=1"

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200 for top_k=1, got %d: %s", rec.Code, rec.Body.String())
	}

	var res inference.PredictSuccessResponse
	json.Unmarshal(rec.Body.Bytes(), &res)
	if len(res.TopPredictions) != 1 {
		t.Errorf("Expected 1 top prediction when top_k=1, got %d", len(res.TopPredictions))
	}
}
