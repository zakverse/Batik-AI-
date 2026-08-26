package routes

import (
	"wastra-ai/backend/internal/config"
	"wastra-ai/backend/internal/handler"
	"wastra-ai/backend/internal/inference"
	"wastra-ai/backend/internal/middleware"
	"wastra-ai/backend/internal/service"

	"github.com/gin-gonic/gin"
)

func SetupRouter(cfg *config.Config, engine *inference.Engine, engine36 *inference.Engine36) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()

	// Middlewares
	router.Use(gin.Logger())
	router.Use(middleware.RecoveryMiddleware())
	router.Use(middleware.CORSMiddleware(cfg.AllowedOrigins))

	// Limit upload size (e.g. 10MB)
	router.MaxMultipartMemory = cfg.MaxUploadSizeMB << 20

	// =========================================================================
	// V1 Services & Handlers (35-Class Legacy)
	// =========================================================================
	if engine != nil {
		predictService := service.NewPredictService(engine)
		healthHandler := handler.NewHealthHandler(engine)
		predictHandler := handler.NewPredictHandler(predictService, cfg.DefaultTopK, cfg.MaxUploadSizeMB)
		benchmarkHandler := handler.NewBenchmarkHandler(predictService)

		// Legacy Root Health Check Endpoint
		router.GET("/health", healthHandler.HealthCheck)

		// API v1 Group (35 Classes)
		v1 := router.Group("/api/v1")
		{
			v1.GET("/health", healthHandler.HealthCheck)
			v1.POST("/predict", predictHandler.Predict)
			v1.POST("/benchmark", benchmarkHandler.Benchmark)
		}
	}

	// =========================================================================
	// V2 Services & Handlers (36-Class Modern: 35 Batik + 1 Non-Batik)
	// =========================================================================
	if engine36 != nil {
		predict36Service := service.NewPredict36Service(engine36)
		health36Handler := handler.NewHealth36Handler(engine36)
		predict36Handler := handler.NewPredict36Handler(predict36Service, cfg.DefaultTopK36, cfg.MaxUploadSizeMB)
		benchmark36Handler := handler.NewBenchmark36Handler(predict36Service)

		// API v2 Group (36 Classes)
		v2 := router.Group("/api/v2")
		{
			v2.GET("/health", health36Handler.HealthCheck)
			v2.POST("/predict", predict36Handler.Predict36)
			v2.POST("/predict/batik", predict36Handler.Predict36) // Alias for explicit domain clarity
			v2.POST("/benchmark", benchmark36Handler.Benchmark36)
		}
	}

	return router
}
