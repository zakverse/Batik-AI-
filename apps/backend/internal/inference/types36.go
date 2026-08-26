package inference

// Model36Info provides model descriptor metadata for API responses
type Model36Info struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Runtime string `json:"runtime"`
}

// Prediction36Item represents an individual prediction item with is_batik flag
type Prediction36Item struct {
	ClassID    int     `json:"class_id"`
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
	IsBatik    bool    `json:"is_batik"`
}

// Predict36SuccessResponse represents the standard mobile-ready 36-class API response
type Predict36SuccessResponse struct {
	Success        bool               `json:"success"`
	Prediction     Prediction36Item   `json:"prediction"`
	TopPredictions []Prediction36Item `json:"top_predictions,omitempty"`
	Model          Model36Info        `json:"model"`
}

// Health36Response represents the v2 health check response
type Health36Response struct {
	Status     string  `json:"status"`
	Model      string  `json:"model"`
	Version    string  `json:"version"`
	Classes    int     `json:"classes"`
	Accuracy   string  `json:"accuracy"`
	MacroF1    float64 `json:"macro_f1"`
	WeightedF1 float64 `json:"weighted_f1"`
	NonBatikF1 float64 `json:"non_batik_f1"`
}

// Benchmark36Response represents the latency benchmarking response for 36-class
type Benchmark36Response struct {
	Success       bool             `json:"success"`
	Model         Model36Info      `json:"model"`
	SamplesCount  int              `json:"samples_count"`
	TotalTimeMs   float64          `json:"total_time_ms"`
	AvgLatencyMs  float64          `json:"avg_latency_ms"`
	P50LatencyMs  float64          `json:"p50_latency_ms"`
	ThroughputFPS float64          `json:"throughput_fps"`
	SamplePred    Prediction36Item `json:"sample_prediction"`
}
