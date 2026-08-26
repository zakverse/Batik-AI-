package service

import (
	"fmt"
	"io"
	"wastra-ai/backend/internal/inference"
)

type Predict36Service struct {
	engine *inference.Engine36
}

func NewPredict36Service(engine *inference.Engine36) *Predict36Service {
	return &Predict36Service{
		engine: engine,
	}
}

func (s *Predict36Service) PredictImage36(r io.Reader, topK int) (*inference.Predict36SuccessResponse, error) {
	// 1. Preprocess Image (RGB 224x224 Bilinear float32 [0..255])
	tensor, err := inference.PreprocessImage(r)
	if err != nil {
		return nil, fmt.Errorf("preprocessing failed: %w", err)
	}

	// 2. Run ONNX Model Inference (36 Classes)
	top1, topKList, err := s.engine.Predict36(tensor, topK)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// 3. Return Standardized Mobile-Ready 36-Class Response
	return &inference.Predict36SuccessResponse{
		Success:        true,
		Prediction:     *top1,
		TopPredictions: topKList,
		Model: inference.Model36Info{
			Name:    "efficientnetb0",
			Version: "36-class",
			Runtime: "onnx",
		},
	}, nil
}

func (s *Predict36Service) GetEngine() *inference.Engine36 {
	return s.engine
}
