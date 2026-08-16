package inference

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

func TestPreprocessImage_Valid(t *testing.T) {
	// Create dummy 100x100 RGB image
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{R: 200, G: 100, B: 50, A: 255})
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("Failed to encode test PNG: %v", err)
	}

	tensor, err := PreprocessImage(&buf)
	if err != nil {
		t.Fatalf("PreprocessImage returned error: %v", err)
	}

	if len(tensor) != TargetTensorSize {
		t.Errorf("Expected tensor size %d, got %d", TargetTensorSize, len(tensor))
	}

	// Verify range [0, 255]
	for i, val := range tensor {
		if val < 0.0 || val > 255.0 {
			t.Errorf("Pixel value at index %d out of bounds [0, 255]: %f", i, val)
			break
		}
	}
}

func TestPreprocessImage_Empty(t *testing.T) {
	var empty bytes.Buffer
	_, err := PreprocessImage(&empty)
	if err == nil {
		t.Errorf("Expected error on empty image, got nil")
	}
}

func TestPreprocessImage_Corrupted(t *testing.T) {
	corrupt := bytes.NewBuffer([]byte("this_is_not_a_valid_image_file"))
	_, err := PreprocessImage(corrupt)
	if err == nil {
		t.Errorf("Expected error on corrupted image, got nil")
	}
}
