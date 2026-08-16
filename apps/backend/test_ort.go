package main

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/image/draw"
)

func toPointer(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

type ortApiBase struct {
	GetApi           uintptr
	GetVersionString uintptr
}

type OrtTestRunner struct {
	dll            *syscall.LazyDLL
	api            *[305]uintptr
	env            uintptr
	sessionOptions uintptr
	session        uintptr
	memInfo        uintptr
	versionString  string
}

func (r *OrtTestRunner) checkStatus(status uintptr) error {
	if status == 0 {
		return nil
	}
	getMsgFn := r.api[2]
	msgPtr, _, _ := syscall.SyscallN(getMsgFn, status)
	var msgBytes []byte
	p := (*byte)(toPointer(msgPtr))
	for p != nil && *p != 0 {
		msgBytes = append(msgBytes, *p)
		p = (*byte)(unsafe.Add(toPointer(uintptr(unsafe.Pointer(p))), 1))
	}
	releaseStatusFn := r.api[93]
	syscall.SyscallN(releaseStatusFn, status)
	return fmt.Errorf("ONNX Runtime Error: %s", string(msgBytes))
}

func preprocessImageFile(imgPath string) ([]float32, error) {
	file, err := os.Open(imgPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image: %w", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %w", err)
	}

	dst := image.NewRGBA(image.Rect(0, 0, 224, 224))
	draw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)

	tensor := make([]float32, 1*224*224*3)
	offset := 0
	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			r, g, b, _ := dst.At(x, y).RGBA()
			tensor[offset] = float32(r >> 8)
			tensor[offset+1] = float32(g >> 8)
			tensor[offset+2] = float32(b >> 8)
			offset += 3
		}
	}
	return tensor, nil
}

func main() {
	fmt.Println("============================================================")
	fmt.Println("🧪 ONNX RUNTIME GO SERVING VALIDATION TEST (test_ort.go)")
	fmt.Println("============================================================")

	baseDir := "."
	if _, err := os.Stat("model"); os.IsNotExist(err) {
		if _, err := os.Stat("apps/backend/model"); err == nil {
			baseDir = "apps/backend"
		}
	}

	dllPath := filepath.Join(baseDir, "model", "onnxruntime.dll")
	modelPath := filepath.Join(baseDir, "model", "efficientnetb0_finetuned.onnx")
	mappingPath := filepath.Join(baseDir, "model", "efficientnetb0_class_mapping.json")

	// 1. Verify DLL exists
	if _, err := os.Stat(dllPath); os.IsNotExist(err) {
		panic(fmt.Sprintf("❌ onnxruntime.dll not found at: %s", dllPath))
	}
	fmt.Println("[PASS] onnxruntime.dll found: " + dllPath)

	// 2. Load DLL and API Base
	dll := syscall.NewLazyDLL(dllPath)
	proc := dll.NewProc("OrtGetApiBase")
	basePtr, _, err := proc.Call()
	if basePtr == 0 {
		panic(fmt.Sprintf("❌ Failed to call OrtGetApiBase: %v", err))
	}

	base := (*ortApiBase)(toPointer(basePtr))
	verPtr, _, _ := syscall.SyscallN(base.GetVersionString)
	var verBytes []byte
	vp := (*byte)(toPointer(verPtr))
	for vp != nil && *vp != 0 {
		verBytes = append(verBytes, *vp)
		vp = (*byte)(unsafe.Add(toPointer(uintptr(unsafe.Pointer(vp))), 1))
	}
	versionStr := string(verBytes)
	fmt.Printf("[PASS] ONNX Runtime Version: %s\n", versionStr)

	apiPtr, _, _ := syscall.SyscallN(base.GetApi, 18) // ORT_API_VERSION 18
	if apiPtr == 0 {
		panic("❌ Failed to get OrtApi pointer for version 18")
	}
	api := (*[305]uintptr)(toPointer(apiPtr))

	runner := &OrtTestRunner{
		dll:           dll,
		api:           api,
		versionString: versionStr,
	}

	// 3. Create Environment
	var env uintptr
	logId, _ := syscall.BytePtrFromString("OrtGoTest")
	status, _, _ := syscall.SyscallN(api[3], 3, uintptr(unsafe.Pointer(logId)), uintptr(unsafe.Pointer(&env)))
	if err := runner.checkStatus(status); err != nil {
		panic(err)
	}
	runner.env = env
	defer syscall.SyscallN(api[92], env)
	fmt.Println("[PASS] ONNX Runtime Environment initialized")

	// 4. Create Session Options
	var sessionOptions uintptr
	status, _, _ = syscall.SyscallN(api[10], uintptr(unsafe.Pointer(&sessionOptions)))
	if err := runner.checkStatus(status); err != nil {
		panic(err)
	}
	runner.sessionOptions = sessionOptions
	defer syscall.SyscallN(api[100], sessionOptions)
	syscall.SyscallN(api[24], sessionOptions, 4)

	// 5. Load Model into Session
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		panic(fmt.Sprintf("❌ Model not found at: %s", modelPath))
	}

	modelPathUTF16, _ := syscall.UTF16PtrFromString(modelPath)
	var session uintptr
	status, _, _ = syscall.SyscallN(api[7], env, uintptr(unsafe.Pointer(modelPathUTF16)), sessionOptions, uintptr(unsafe.Pointer(&session)))
	if err := runner.checkStatus(status); err != nil {
		panic(err)
	}
	runner.session = session
	defer syscall.SyscallN(api[95], session)
	fmt.Printf("[PASS] Model loaded successfully: %s\n", modelPath)

	// 6. Create Memory Info
	var memInfo uintptr
	status, _, _ = syscall.SyscallN(api[69], 0, 0, uintptr(unsafe.Pointer(&memInfo)))
	if err := runner.checkStatus(status); err != nil {
		panic(err)
	}
	runner.memInfo = memInfo
	defer syscall.SyscallN(api[94], memInfo)

	// 7. Load Class Mapping
	mapData, err := os.ReadFile(mappingPath)
	if err != nil {
		panic(err)
	}
	var rawMapping map[string]string
	json.Unmarshal(mapData, &rawMapping)
	if len(rawMapping) != 35 {
		panic(fmt.Sprintf("❌ Expected 35 classes, found %d", len(rawMapping)))
	}
	classMap := make(map[int]string, 35)
	for k, v := range rawMapping {
		var idx int
		fmt.Sscanf(k, "%d", &idx)
		classMap[idx] = v
	}
	fmt.Printf("[PASS] Class Mapping loaded: %d classes verified\n", len(classMap))

	// Model specifications
	fmt.Println("------------------------------------------------------------")
	fmt.Println("MODEL SPECIFICATIONS:")
	fmt.Println("  • Input Name  : input_1")
	fmt.Println("  • Input Shape : [1, 224, 224, 3] (float32)")
	fmt.Println("  • Output Name : predictions")
	fmt.Println("  • Output Shape: [1, 35] (float32 probabilities)")
	fmt.Println("------------------------------------------------------------")

	// 8. Find a real test sample image
	testSamplePath := ""
	candidatePaths := []string{
		"../../datasets/raw/dataset_augmented/batik-bali/aug_0_2655.jpeg",
		"datasets/raw/dataset_augmented/batik-bali/aug_0_2655.jpeg",
		"../../datasets/raw/dataset_augmented/Bali_Barong/aug_0_2352.jpeg",
		"datasets/raw/dataset_augmented/Bali_Barong/aug_0_2352.jpeg",
	}
	for _, cp := range candidatePaths {
		if _, err := os.Stat(cp); err == nil {
			testSamplePath = cp
			break
		}
	}

	var inputFloats []float32
	if testSamplePath != "" {
		fmt.Printf("🔍 Running real inference on test sample: %s\n", testSamplePath)
		inputFloats, err = preprocessImageFile(testSamplePath)
		if err != nil {
			panic(err)
		}
	} else {
		fmt.Println("⚠️ Test sample image not found, generating deterministic test tensor...")
		inputFloats = make([]float32, 1*224*224*3)
		for i := range inputFloats {
			inputFloats[i] = float32(i%256) / 255.0
		}
	}

	// 9. Execute Inference via ONNX Runtime
	inputShape := []int64{1, 224, 224, 3}
	var inputTensor uintptr
	status, _, _ = syscall.SyscallN(api[49],
		memInfo,
		uintptr(unsafe.Pointer(&inputFloats[0])),
		uintptr(len(inputFloats)*4),
		uintptr(unsafe.Pointer(&inputShape[0])),
		4,
		1, // FLOAT
		uintptr(unsafe.Pointer(&inputTensor)),
	)
	if err := runner.checkStatus(status); err != nil {
		panic(err)
	}
	defer syscall.SyscallN(api[96], inputTensor)

	inputNameBytes, _ := syscall.BytePtrFromString("input_1")
	outputNameBytes, _ := syscall.BytePtrFromString("predictions")
	inputNames := []uintptr{uintptr(unsafe.Pointer(inputNameBytes))}
	outputNames := []uintptr{uintptr(unsafe.Pointer(outputNameBytes))}
	inputs := []uintptr{inputTensor}
	outputs := []uintptr{0}

	t0 := time.Now()
	status, _, _ = syscall.SyscallN(api[9],
		session,
		0,
		uintptr(unsafe.Pointer(&inputNames[0])),
		uintptr(unsafe.Pointer(&inputs[0])),
		1,
		uintptr(unsafe.Pointer(&outputNames[0])),
		1,
		uintptr(unsafe.Pointer(&outputs[0])),
	)
	if err := runner.checkStatus(status); err != nil {
		panic(err)
	}
	infDuration := time.Since(t0)
	outputTensor := outputs[0]
	defer syscall.SyscallN(api[96], outputTensor)

	var outData unsafe.Pointer
	status, _, _ = syscall.SyscallN(api[51], outputTensor, uintptr(unsafe.Pointer(&outData)))
	if err := runner.checkStatus(status); err != nil {
		panic(err)
	}

	rawProbs := unsafe.Slice((*float32)(outData), 35)

	// Softmax Calculation
	probs := make([]float64, 35)
	sum := 0.0
	for _, v := range rawProbs {
		sum += float64(v)
	}

	if sum > 1.05 || sum < 0.95 {
		maxVal := -math.MaxFloat64
		for _, v := range rawProbs {
			if float64(v) > maxVal {
				maxVal = float64(v)
			}
		}
		expSum := 0.0
		for i, v := range rawProbs {
			probs[i] = math.Exp(float64(v) - maxVal)
			expSum += probs[i]
		}
		for i := range probs {
			probs[i] /= expSum
		}
	} else {
		for i, v := range rawProbs {
			probs[i] = float64(v)
		}
	}

	// 10. Display 35-Class Probabilities & Top Predictions
	type indexedProb struct {
		Index int
		Class string
		Prob  float64
	}
	allProbs := make([]indexedProb, 35)
	for i, p := range probs {
		allProbs[i] = indexedProb{Index: i, Class: classMap[i], Prob: p}
	}

	sort.Slice(allProbs, func(i, j int) bool {
		return allProbs[i].Prob > allProbs[j].Prob
	})

	fmt.Println("\n📊 35-CLASS INFERENCE PREDICTION OUTPUT:")
	fmt.Println("------------------------------------------------------------")
	for i := 0; i < 35; i++ {
		fmt.Printf("  [%2d] %-32s : %8.4f%%\n", i, classMap[i], probs[i]*100)
	}
	fmt.Println("------------------------------------------------------------")

	fmt.Println("\n🏆 TOP 3 PREDICTIONS:")
	for i := 0; i < 3; i++ {
		fmt.Printf("  %d. %-28s (Confidence: %.4f / %.2f%%)\n", i+1, allProbs[i].Class, allProbs[i].Prob, allProbs[i].Prob*100)
	}
	fmt.Printf("⏱️ Inference Latency: %.2f ms (%.2f FPS)\n", float64(infDuration.Microseconds())/1000.0, 1.0/infDuration.Seconds())

	fmt.Println("\n============================================================")
	fmt.Println("FINAL TEST STATUS: PASS [OK]")
	fmt.Println("============================================================")
}
