package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
	"wastra-ai/backend/internal/config"
	"wastra-ai/backend/internal/inference"
	"wastra-ai/backend/routes"
)

func main() {
	fmt.Println("============================================================")
	fmt.Println("🚀 WASTRA AI BATIK — GOLANG ONNX INFERENCE BACKEND")
	fmt.Println("============================================================")

	// 1. Load Config
	cfg := config.LoadConfig()
	fmt.Println("[PASS] Config loaded successfully")

	// 2. Startup Validation & Engine Initialization
	var engine *inference.Engine
	var engine36 *inference.Engine36

	// 2a. Initialize 35-Class Legacy Engine (if model exists)
	if _, err := os.Stat(cfg.ModelPath); err == nil {
		e, err := inference.NewEngine(
			cfg.ModelPath,
			cfg.ClassMappingPath,
			cfg.MetadataPath,
			cfg.ONNXLibPath,
		)
		if err != nil {
			log.Printf("⚠️ WARNING: Legacy 35-class engine initialization failed: %v", err)
		} else {
			engine = e
			defer engine.Close()
			meta := engine.GetMetadata()
			fmt.Println("[PASS] 35-Class Legacy Model loaded: " + cfg.ModelPath)
			fmt.Printf("[PASS] 35-Class Metadata: %s (Test Acc: %.2f%%, Macro F1: %.4f)\n", meta.ModelName, meta.TestAccuracy*100, meta.MacroF1)
		}
	}

	// 2b. Initialize 36-Class Modern Engine
	e36, err := inference.NewEngine36(
		cfg.Model36Path,
		cfg.ClassMapping36Path,
		cfg.Metadata36Path,
		cfg.ONNXLibPath,
	)
	if err != nil {
		log.Fatalf("❌ STARTUP ERROR: Failed to initialize 36-Class ONNX Engine:\n   %v\n", err)
	}
	engine36 = e36
	defer engine36.Close()

	fmt.Println("[PASS] 36-Class Modern Model loaded: " + cfg.Model36Path)
	fmt.Printf("[PASS] 36-Class Classes: %d classes (Class 35 = non_batik)\n", engine36.GetClassCount())
	fmt.Println("[PASS] ONNX Runtime initialized with shared library: " + cfg.ONNXLibPath)
	fmt.Println("[PASS] Input shape verified: 224x224x3 (RGB float32)")
	fmt.Println("[PASS] Output shape verified: 36 classes")
	fmt.Println("============================================================")

	// 3. Setup Routes (Dual Engine Architecture)
	router := routes.SetupRouter(cfg, engine, engine36)

	addr := fmt.Sprintf("%s:%s", cfg.Host, cfg.Port)
	srv := &http.Server{
		Addr:         addr,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// 4. Start Server in background
	go func() {
		fmt.Printf("🌐 Server listening on http://%s\n", addr)
		fmt.Println("Available Endpoints:")
		fmt.Println("  • GET  /health              (Legacy 35-Class Health)")
		fmt.Println("  • POST /api/v1/predict      (Legacy 35-Class Prediction)")
		fmt.Println("  • POST /api/v1/benchmark    (Legacy 35-Class Benchmark)")
		fmt.Println("  • GET  /api/v2/health       (Modern 36-Class Health)")
		fmt.Println("  • POST /api/v2/predict      (Modern 36-Class Prediction)")
		fmt.Println("  • POST /api/v2/predict/batik(Modern 36-Class Prediction Alias)")
		fmt.Println("  • POST /api/v2/benchmark    (Modern 36-Class Benchmark)")
		fmt.Println("============================================================")

		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server listen error: %v", err)
		}
	}()

	// 5. Graceful Shutdown Listener
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	fmt.Println("\nShutting down server gracefully...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("Server forced to shutdown: %v", err)
	}
	fmt.Println("Server exited cleanly.")
}
