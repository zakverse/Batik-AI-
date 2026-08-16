package inference

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"

	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

const (
	TargetImageWidth  = 224
	TargetImageHeight = 224
	TargetChannels    = 3
	TargetTensorSize  = TargetImageWidth * TargetImageHeight * TargetChannels
)

// PreprocessImage decodes, converts to RGB, resizes to 224x224, and produces a float32 tensor
// matching Keras / EfficientNetB0 standard preprocessing [0, 255] float range in RGB order.
func PreprocessImage(r io.Reader) ([]float32, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read image data: %w", err)
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("empty image data provided")
	}

	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to decode image format: %w", err)
	}

	// Create 224x224 RGBA canvas
	dst := image.NewRGBA(image.Rect(0, 0, TargetImageWidth, TargetImageHeight))

	// High-quality Bilinear Resizing
	draw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)

	// Convert to flat float32 tensor in (1, 224, 224, 3) format: [R0, G0, B0, R1, G1, B1, ...]
	// Values in range [0.0, 255.0] float32 matching tf.keras.applications.efficientnet.preprocess_input
	tensor := make([]float32, TargetTensorSize)
	offset := 0

	for y := 0; y < TargetImageHeight; y++ {
		for x := 0; x < TargetImageWidth; x++ {
			r, g, b, _ := dst.At(x, y).RGBA()
			// RGBA() returns [0, 65535], convert back to [0, 255]
			tensor[offset] = float32(r >> 8)
			tensor[offset+1] = float32(g >> 8)
			tensor[offset+2] = float32(b >> 8)
			offset += 3
		}
	}

	return tensor, nil
}
