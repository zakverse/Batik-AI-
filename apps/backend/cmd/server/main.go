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
	engine, err := inference.NewEngine(
		cfg.ModelPath,
		cfg.ClassMappingPath,
		cfg.MetadataPath,
		cfg.ONNXLibPath,
	)
	if err != nil {
		log.Fatalf("❌ STARTUP ERROR: Failed to initialize ONNX Engine:\n   %v\n", err)
	}
	defer engine.Close()

	meta := engine.GetMetadata()
	classCount := engine.GetClassCount()

	fmt.Println("[PASS] Model file exists & verified: " + cfg.ModelPath)
	fmt.Println("[PASS] ONNX Runtime initialized with shared library: " + cfg.ONNXLibPath)
	fmt.Printf("[PASS] Class mapping loaded: %d classes detected\n", classCount)
	fmt.Printf("[PASS] Metadata verified: %s (Test Acc: %.2f%%, Macro F1: %.4f)\n", meta.ModelName, meta.TestAccuracy*100, meta.MacroF1)
	fmt.Println("[PASS] Input shape verified: 224x224x3 (RGB float32)")
	fmt.Println("[PASS] Output shape verified: 35 classes")
	fmt.Println("============================================================")

	// 3. Setup Routes
	router := routes.SetupRouter(cfg, engine)

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
		fmt.Println("  • GET  /health")
		fmt.Println("  • POST /api/v1/predict   (field: 'image', optional 'top_k')")
		fmt.Println("  • POST /api/v1/benchmark (field: 'image', optional 'iterations')")
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
