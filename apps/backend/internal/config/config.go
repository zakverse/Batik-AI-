package config

import (
	"os"
	"path/filepath"
	"strconv"
)

type Config struct {
	Port             string
	Host             string
	ModelPath        string
	ClassMappingPath string
	MetadataPath     string
	ONNXLibPath      string
	MaxUploadSizeMB  int64
	DefaultTopK      int
	AllowedOrigins   string
}

func LoadConfig() *Config {
	port := getEnv("PORT", "8080")
	host := getEnv("HOST", "0.0.0.0")

	// Determine base directory relative to working directory or executable
	baseDir := "."
	if _, err := os.Stat("model"); os.IsNotExist(err) {
		if _, err := os.Stat("apps/backend/model"); err == nil {
			baseDir = "apps/backend"
		}
	}

	modelPath := getEnv("MODEL_PATH", filepath.Join(baseDir, "model", "efficientnetb0_finetuned.onnx"))
	classMappingPath := getEnv("CLASS_MAPPING_PATH", filepath.Join(baseDir, "model", "efficientnetb0_class_mapping.json"))
	metadataPath := getEnv("METADATA_PATH", filepath.Join(baseDir, "model", "efficientnetb0_model_metadata.json"))

	// Support both ONNX_RUNTIME_PATH and ONNX_LIB_PATH
	onnxLibDefault := filepath.Join(baseDir, "model", "onnxruntime.dll")
	onnxLibPath := getEnv("ONNX_RUNTIME_PATH", getEnv("ONNX_LIB_PATH", onnxLibDefault))

	maxUploadMB, _ := strconv.ParseInt(getEnv("MAX_UPLOAD_SIZE_MB", "10"), 10, 64)
	if maxUploadMB <= 0 {
		maxUploadMB = 10
	}

	defaultTopK, _ := strconv.Atoi(getEnv("DEFAULT_TOP_K", "3"))
	if defaultTopK <= 0 || defaultTopK > 35 {
		defaultTopK = 3
	}

	allowedOrigins := getEnv("ALLOWED_ORIGINS", "*")

	return &Config{
		Port:             port,
		Host:             host,
		ModelPath:        modelPath,
		ClassMappingPath: classMappingPath,
		MetadataPath:     metadataPath,
		ONNXLibPath:      onnxLibPath,
		MaxUploadSizeMB:  maxUploadMB,
		DefaultTopK:      defaultTopK,
		AllowedOrigins:   allowedOrigins,
	}
}

func getEnv(key, fallback string) string {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		return val
	}
	return fallback
}
