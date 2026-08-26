package config

import (
	"os"
	"path/filepath"
	"strconv"
)

type Config struct {
	Port               string
	Host               string
	ModelPath          string
	ClassMappingPath   string
	MetadataPath       string
	Model36Path        string
	ClassMapping36Path string
	Metadata36Path     string
	ONNXLibPath        string
	MaxUploadSizeMB    int64
	DefaultTopK        int
	DefaultTopK36      int
	AllowedOrigins     string
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

	// 35-Class Legacy Configuration
	modelPath := getEnv("MODEL_PATH", filepath.Join(baseDir, "model", "efficientnetb0_finetuned.onnx"))
	classMappingPath := getEnv("CLASS_MAPPING_PATH", filepath.Join(baseDir, "model", "efficientnetb0_class_mapping.json"))
	metadataPath := getEnv("METADATA_PATH", filepath.Join(baseDir, "model", "efficientnetb0_model_metadata.json"))

	// 36-Class Modern Configuration
	model36Path := getEnv("MODEL_36CLASS_PATH", filepath.Join(baseDir, "model", "efficientnetb0_36class_finetuned.onnx"))
	classMapping36Path := getEnv("CLASS_MAPPING_36CLASS_PATH", filepath.Join(baseDir, "model", "efficientnetb0_36class_class_mapping.json"))
	metadata36Path := getEnv("METADATA_36CLASS_PATH", filepath.Join(baseDir, "model", "efficientnetb0_36class_model_metadata.json"))

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

	defaultTopK36, _ := strconv.Atoi(getEnv("DEFAULT_TOP_K_36", "3"))
	if defaultTopK36 <= 0 || defaultTopK36 > 36 {
		defaultTopK36 = 3
	}

	allowedOrigins := getEnv("ALLOWED_ORIGINS", "*")

	return &Config{
		Port:               port,
		Host:               host,
		ModelPath:          modelPath,
		ClassMappingPath:   classMappingPath,
		MetadataPath:       metadataPath,
		Model36Path:        model36Path,
		ClassMapping36Path: classMapping36Path,
		Metadata36Path:     metadata36Path,
		ONNXLibPath:        onnxLibPath,
		MaxUploadSizeMB:    maxUploadMB,
		DefaultTopK:        defaultTopK,
		DefaultTopK36:      defaultTopK36,
		AllowedOrigins:     allowedOrigins,
	}
}

func getEnv(key, fallback string) string {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		return val
	}
	return fallback
}
