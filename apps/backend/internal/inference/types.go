package inference

type ModelMetadata struct {
	ModelName      string    `json:"model_name"`
	Architecture   string    `json:"architecture"`
	Pretrained     string    `json:"pretrained"`
	InputSize      [2]int    `json:"input_size"`
	Channels       int       `json:"channels"`
	NumClasses     int       `json:"num_classes"`
	TrainingStage  string    `json:"training_stage"`
	UnfrozenLayers int       `json:"unfrozen_layers"`
	LearningRate   float64   `json:"learning_rate"`
	Optimizer      string    `json:"optimizer"`
	Loss           string    `json:"loss"`
	TestAccuracy   float64   `json:"test_accuracy"`
	MacroF1        float64   `json:"macro_f1"`
	WeightedF1     float64   `json:"weighted_f1"`
}

type PredictionItem struct {
	Class      string  `json:"class"`
	Confidence float64 `json:"confidence"`
}

type PredictSuccessResponse struct {
	Success        bool             `json:"success"`
	Prediction     PredictionItem   `json:"prediction"`
	TopPredictions []PredictionItem `json:"top_predictions"`
}

type APIErrorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type ErrorResponse struct {
	Success bool           `json:"success"`
	Error   APIErrorDetail `json:"error"`
}

type HealthResponse struct {
	Status     string `json:"status"`
	Model      string `json:"model"`
	Classes    int    `json:"classes"`
	Version    string `json:"version"`
	Accuracy   string `json:"test_accuracy"`
	Generalize string `json:"generalization_gap"`
}

type BenchmarkResponse struct {
	Success        bool    `json:"success"`
	SamplesCount   int     `json:"samples_count"`
	TotalTimeMs    float64 `json:"total_time_ms"`
	AvgLatencyMs   float64 `json:"avg_latency_ms"`
	ThroughputFPS  float64 `json:"throughput_fps"`
	PredictedClass string  `json:"sample_predicted_class"`
	Confidence     float64 `json:"sample_confidence"`
}
