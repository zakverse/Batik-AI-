package inference

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"syscall"
	"unsafe"
)

type Engine struct {
	mu             sync.Mutex
	dll            *syscall.LazyDLL
	api            *[305]uintptr
	env            uintptr
	sessionOptions uintptr
	session        uintptr
	memInfo        uintptr
	classMapping   map[int]string
	metadata       ModelMetadata
	isClosed       bool
}

func (e *Engine) checkStatus(status uintptr) error {
	if status == 0 {
		return nil
	}
	getMsgFn := e.api[2]
	msgPtr, _, _ := syscall.SyscallN(getMsgFn, status)
	var msgBytes []byte
	p := (*byte)(unsafe.Pointer(msgPtr))
	for *p != 0 {
		msgBytes = append(msgBytes, *p)
		p = (*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + 1))
	}
	releaseStatusFn := e.api[93]
	syscall.SyscallN(releaseStatusFn, status)
	return fmt.Errorf("ONNX Runtime Error: %s", string(msgBytes))
}

func NewEngine(modelPath, classMappingPath, metadataPath, dllPath string) (*Engine, error) {
	// 1. Verify Model Exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model file not found at: %s", modelPath)
	}

	// 2. Load Class Mapping
	mapData, err := os.ReadFile(classMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read class mapping at %s: %w", classMappingPath, err)
	}

	var rawMap map[string]string
	if err := json.Unmarshal(mapData, &rawMap); err != nil {
		return nil, fmt.Errorf("failed to parse class mapping JSON: %w", err)
	}

	if len(rawMap) != 35 {
		return nil, fmt.Errorf("expected 35 classes in class mapping, got %d", len(rawMap))
	}

	classMapping := make(map[int]string, 35)
	for k, v := range rawMap {
		var idx int
		fmt.Sscanf(k, "%d", &idx)
		classMapping[idx] = v
	}

	// 3. Load Model Metadata
	metaData, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model metadata at %s: %w", metadataPath, err)
	}

	var metadata ModelMetadata
	if err := json.Unmarshal(metaData, &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse model metadata JSON: %w", err)
	}

	// 4. Load ONNX DLL
	if _, err := os.Stat(dllPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("onnxruntime.dll not found at: %s", dllPath)
	}

	dll := syscall.NewLazyDLL(dllPath)
	proc := dll.NewProc("OrtGetApiBase")
	basePtr, _, err := proc.Call()
	if basePtr == 0 {
		return nil, fmt.Errorf("failed to call OrtGetApiBase: %v", err)
	}

	getApiFn := *(*uintptr)(unsafe.Pointer(basePtr))
	apiPtr, _, _ := syscall.SyscallN(getApiFn, 18) // ORT_API_VERSION 18
	if apiPtr == 0 {
		return nil, fmt.Errorf("failed to get OrtApi pointer (version 18)")
	}

	api := (*[305]uintptr)(unsafe.Pointer(apiPtr))
	engine := &Engine{
		dll:          dll,
		api:          api,
		classMapping: classMapping,
		metadata:     metadata,
	}

	// 5. Create Env (index 3)
	var env uintptr
	logId, _ := syscall.BytePtrFromString("WastraAIBatikGoServer")
	status, _, _ := syscall.SyscallN(api[3], 3, uintptr(unsafe.Pointer(logId)), uintptr(unsafe.Pointer(&env)))
	if err := engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to create OrtEnv: %w", err)
	}
	engine.env = env

	// 6. Create SessionOptions (index 10)
	var sessionOptions uintptr
	status, _, _ = syscall.SyscallN(api[10], uintptr(unsafe.Pointer(&sessionOptions)))
	if err := engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to create OrtSessionOptions: %w", err)
	}
	engine.sessionOptions = sessionOptions

	// Set IntraOpNumThreads = 4 (index 24)
	syscall.SyscallN(api[24], sessionOptions, 4)

	// 7. Create Session (index 7)
	modelPathUTF16, err := syscall.UTF16PtrFromString(modelPath)
	if err != nil {
		return nil, fmt.Errorf("invalid model path encoding: %w", err)
	}
	var session uintptr
	status, _, _ = syscall.SyscallN(api[7], env, uintptr(unsafe.Pointer(modelPathUTF16)), sessionOptions, uintptr(unsafe.Pointer(&session)))
	if err := engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to create OrtSession: %w", err)
	}
	engine.session = session

	// 8. Create CpuMemoryInfo (index 69)
	var memInfo uintptr
	status, _, _ = syscall.SyscallN(api[69], 0, 0, uintptr(unsafe.Pointer(&memInfo)))
	if err := engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to create OrtMemoryInfo: %w", err)
	}
	engine.memInfo = memInfo

	return engine, nil
}

func (e *Engine) Predict(inputFloats []float32, topK int) (*PredictionItem, []PredictionItem, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.isClosed {
		return nil, nil, fmt.Errorf("inference engine is closed")
	}

	if len(inputFloats) != TargetTensorSize {
		return nil, nil, fmt.Errorf("invalid input tensor size: expected %d, got %d", TargetTensorSize, len(inputFloats))
	}

	if topK <= 0 {
		topK = 3
	}
	if topK > 35 {
		topK = 35
	}

	// 1. Create Input Tensor (index 49: CreateTensorWithDataAsOrtValue)
	inputShape := []int64{1, TargetImageHeight, TargetImageWidth, TargetChannels}
	var inputTensor uintptr
	status, _, _ := syscall.SyscallN(e.api[49],
		e.memInfo,
		uintptr(unsafe.Pointer(&inputFloats[0])),
		uintptr(len(inputFloats)*4),
		uintptr(unsafe.Pointer(&inputShape[0])),
		4,
		1, // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
		uintptr(unsafe.Pointer(&inputTensor)),
	)
	if err := e.checkStatus(status); err != nil {
		return nil, nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer syscall.SyscallN(e.api[96], inputTensor) // ReleaseValue

	// 2. Run Inference (index 9: Run)
	inputNameBytes, _ := syscall.BytePtrFromString("input_1")
	outputNameBytes, _ := syscall.BytePtrFromString("predictions")
	inputNames := []uintptr{uintptr(unsafe.Pointer(inputNameBytes))}
	outputNames := []uintptr{uintptr(unsafe.Pointer(outputNameBytes))}
	inputs := []uintptr{inputTensor}
	outputs := []uintptr{0}

	status, _, _ = syscall.SyscallN(e.api[9],
		e.session,
		0,
		uintptr(unsafe.Pointer(&inputNames[0])),
		uintptr(unsafe.Pointer(&inputs[0])),
		1,
		uintptr(unsafe.Pointer(&outputNames[0])),
		1,
		uintptr(unsafe.Pointer(&outputs[0])),
	)
	if err := e.checkStatus(status); err != nil {
		return nil, nil, fmt.Errorf("failed to run inference session: %w", err)
	}
	outputTensor := outputs[0]
	defer syscall.SyscallN(e.api[96], outputTensor) // ReleaseValue

	// 3. Extract Output Data (index 51: GetTensorMutableData)
	var outDataPtr uintptr
	status, _, _ = syscall.SyscallN(e.api[51], outputTensor, uintptr(unsafe.Pointer(&outDataPtr)))
	if err := e.checkStatus(status); err != nil {
		return nil, nil, fmt.Errorf("failed to get output tensor data: %w", err)
	}

	rawProbs := unsafe.Slice((*float32)(unsafe.Pointer(outDataPtr)), 35)

	// 4. Compute Softmax Probabilities
	sum := 0.0
	for _, v := range rawProbs {
		sum += float64(v)
	}

	probs := make([]float64, 35)
	if sum > 1.05 || sum < 0.95 {
		// Logits detected: apply numerical stable softmax
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

	// 5. Build Top-K Ranking
	type indexedProb struct {
		Index int
		Prob  float64
	}
	indexed := make([]indexedProb, 35)
	for i, p := range probs {
		indexed[i] = indexedProb{Index: i, Prob: p}
	}

	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].Prob > indexed[j].Prob
	})

	topPredictions := make([]PredictionItem, topK)
	for i := 0; i < topK; i++ {
		className := e.classMapping[indexed[i].Index]
		topPredictions[i] = PredictionItem{
			Class:      className,
			Confidence: math.Round(indexed[i].Prob*10000) / 10000,
		}
	}

	top1 := &topPredictions[0]
	return top1, topPredictions, nil
}

func (e *Engine) GetMetadata() ModelMetadata {
	return e.metadata
}

func (e *Engine) GetClassCount() int {
	return len(e.classMapping)
}

func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.isClosed {
		if e.memInfo != 0 {
			syscall.SyscallN(e.api[94], e.memInfo) // ReleaseMemoryInfo
		}
		if e.session != 0 {
			syscall.SyscallN(e.api[95], e.session) // ReleaseSession
		}
		if e.sessionOptions != 0 {
			syscall.SyscallN(e.api[100], e.sessionOptions) // ReleaseSessionOptions
		}
		if e.env != 0 {
			syscall.SyscallN(e.api[92], e.env) // ReleaseEnv
		}
		e.isClosed = true
	}
}
