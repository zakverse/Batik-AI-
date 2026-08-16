package service

import (
	"fmt"
	"io"
	"wastra-ai/backend/internal/inference"
)

type PredictService struct {
	engine *inference.Engine
}

func NewPredictService(engine *inference.Engine) *PredictService {
	return &PredictService{
		engine: engine,
	}
}

func (s *PredictService) PredictImage(r io.Reader, topK int) (*inference.PredictSuccessResponse, error) {
	// 1. Preprocess Image
	tensor, err := inference.PreprocessImage(r)
	if err != nil {
		return nil, fmt.Errorf("preprocessing failed: %w", err)
	}

	// 2. Run ONNX Model Inference
	top1, topKList, err := s.engine.Predict(tensor, topK)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// 3. Return Success Response
	return &inference.PredictSuccessResponse{
		Success:        true,
		Prediction:     *top1,
		TopPredictions: topKList,
	}, nil
}

func (s *PredictService) GetEngine() *inference.Engine {
	return s.engine
}
