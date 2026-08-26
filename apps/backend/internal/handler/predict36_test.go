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

func find36ModelDir() string {
	candidates := []string{
		"model",
		"../model",
		"../../model",
		"../../../apps/backend/model",
		"apps/backend/model",
	}
	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(c, "efficientnetb0_36class_finetuned.onnx")); err == nil {
			return c
		}
	}
	return "model"
}

func setupTest36Router(t *testing.T) (*gin.Engine, *inference.Engine36) {
	gin.SetMode(gin.TestMode)
	modelDir := find36ModelDir()

	cfg := &config.Config{
		Port:               "8080",
		Host:               "0.0.0.0",
		Model36Path:        filepath.Join(modelDir, "efficientnetb0_36class_finetuned.onnx"),
		ClassMapping36Path: filepath.Join(modelDir, "efficientnetb0_36class_class_mapping.json"),
		Metadata36Path:     filepath.Join(modelDir, "efficientnetb0_36class_model_metadata.json"),
		ONNXLibPath:        filepath.Join(modelDir, "onnxruntime.dll"),
		MaxUploadSizeMB:    10,
		DefaultTopK36:      3,
		AllowedOrigins:     "*",
	}

	engine36, err := inference.NewEngine36(cfg.Model36Path, cfg.ClassMapping36Path, cfg.Metadata36Path, cfg.ONNXLibPath)
	if err != nil {
		t.Fatalf("Failed to initialize 36-class test engine: %v", err)
	}

	router := routes.SetupRouter(cfg, nil, engine36)
	return router, engine36
}

func createMultipartRequest36(url, fieldName, filename string, fileContent []byte) (*http.Request, string, error) {
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

	req := httptest.NewRequest(http.MethodPost, url, &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return req, writer.FormDataContentType(), nil
}

func generate36TestJPEG() []byte {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{R: 210, G: 120, B: 60, A: 255})
		}
	}
	var buf bytes.Buffer
	_ = jpeg.Encode(&buf, img, nil)
	return buf.Bytes()
}

func TestHealth36Endpoint(t *testing.T) {
	router, engine36 := setupTest36Router(t)
	defer engine36.Close()

	req := httptest.NewRequest(http.MethodGet, "/api/v2/health", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", rec.Code)
	}

	var res inference.Health36Response
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatalf("Failed to unmarshal 36-class health response: %v", err)
	}

	if res.Status != "ok" || res.Classes != 36 {
		t.Errorf("Unexpected health response: %+v", res)
	}
}

func TestPredict36_NoImage(t *testing.T) {
	router, engine36 := setupTest36Router(t)
	defer engine36.Close()

	req, _, _ := createMultipartRequest36("/api/v2/predict", "", "", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for missing image, got %d", rec.Code)
	}
}

func TestPredict36_InvalidTopK(t *testing.T) {
	router, engine36 := setupTest36Router(t)
	defer engine36.Close()

	validJPEG := generate36TestJPEG()
	req, _, _ := createMultipartRequest36("/api/v2/predict", "image", "batik.jpg", validJPEG)
	req.URL.RawQuery = "top_k=50" // Exceeds max 36

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400 for top_k > 36, got %d", rec.Code)
	}
}

func TestPredict36_ValidImage_Success(t *testing.T) {
	router, engine36 := setupTest36Router(t)
	defer engine36.Close()

	validJPEG := generate36TestJPEG()
	req, _, _ := createMultipartRequest36("/api/v2/predict", "image", "batik_sample.jpg", validJPEG)

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200 for valid image, got %d: %s", rec.Code, rec.Body.String())
	}

	var res inference.Predict36SuccessResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &res); err != nil {
		t.Fatalf("Failed to decode 36-class success response: %v", err)
	}

	if !res.Success {
		t.Errorf("Expected success=true, got %v", res.Success)
	}

	if res.Prediction.Label == "" || res.Prediction.ClassID < 0 || res.Prediction.ClassID > 35 {
		t.Errorf("Invalid prediction item: %+v", res.Prediction)
	}

	if res.Model.Version != "36-class" {
		t.Errorf("Expected model version '36-class', got %s", res.Model.Version)
	}
}

func TestPredict36_AliasEndpoint(t *testing.T) {
	router, engine36 := setupTest36Router(t)
	defer engine36.Close()

	validJPEG := generate36TestJPEG()
	req, _, _ := createMultipartRequest36("/api/v2/predict/batik", "image", "batik_sample.jpg", validJPEG)

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("Expected status 200 for alias endpoint, got %d: %s", rec.Code, rec.Body.String())
	}
}
