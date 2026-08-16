package routes

import (
	"wastra-ai/backend/internal/config"
	"wastra-ai/backend/internal/handler"
	"wastra-ai/backend/internal/inference"
	"wastra-ai/backend/internal/middleware"
	"wastra-ai/backend/internal/service"

	"github.com/gin-gonic/gin"
)

func SetupRouter(cfg *config.Config, engine *inference.Engine) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()

	// Middlewares
	router.Use(gin.Logger())
	router.Use(middleware.RecoveryMiddleware())
	router.Use(middleware.CORSMiddleware(cfg.AllowedOrigins))

	// Limit upload size (e.g. 10MB)
	router.MaxMultipartMemory = cfg.MaxUploadSizeMB << 20

	// Services & Handlers
	predictService := service.NewPredictService(engine)
	healthHandler := handler.NewHealthHandler(engine)
	predictHandler := handler.NewPredictHandler(predictService, cfg.DefaultTopK, cfg.MaxUploadSizeMB)
	benchmarkHandler := handler.NewBenchmarkHandler(predictService)

	// Health Check Endpoint
	router.GET("/health", healthHandler.HealthCheck)

	// API v1 Group
	v1 := router.Group("/api/v1")
	{
		v1.POST("/predict", predictHandler.Predict)
		v1.POST("/benchmark", benchmarkHandler.Benchmark)
	}

	return router
}
