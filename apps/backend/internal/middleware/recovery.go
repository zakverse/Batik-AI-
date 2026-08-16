package middleware

import (
	"fmt"
	"net/http"
	"wastra-ai/backend/internal/inference"

	"github.com/gin-gonic/gin"
)

func RecoveryMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		defer func() {
			if r := recover(); r != nil {
				c.JSON(http.StatusInternalServerError, inference.ErrorResponse{
					Success: false,
					Error: inference.APIErrorDetail{
						Code:    "INTERNAL_SERVER_ERROR",
						Message: fmt.Sprintf("Server encountered an unexpected error: %v", r),
					},
				})
				c.Abort()
			}
		}()
		c.Next()
	}
}
