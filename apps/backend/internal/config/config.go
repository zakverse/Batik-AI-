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
	onnxLibPath := getEnv("ONNX_LIB_PATH", filepath.Join(baseDir, "model", "onnxruntime.dll"))

	maxUploadMB, _ := strconv.ParseInt(getEnv("MAX_UPLOAD_SIZE_MB", "10"), 10, 64)
	defaultTopK, _ := strconv.Atoi(getEnv("DEFAULT_TOP_K", "3"))

	return &Config{
		Port:             port,
		Host:             host,
		ModelPath:        modelPath,
		ClassMappingPath: classMappingPath,
		MetadataPath:     metadataPath,
		ONNXLibPath:      onnxLibPath,
		MaxUploadSizeMB:  maxUploadMB,
		DefaultTopK:      defaultTopK,
	}
}

func getEnv(key, fallback string) string {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		return val
	}
	return fallback
}
