package scan

import "time"

type ScanHistory struct {
	ID         string    `json:"id" gorm:"primaryKey"`
	UserID     uint      `json:"user_id"`
	MotifID    uint      `json:"motif_id"`
	Confidence float64   `json:"confidence"`
	ImagePath  string    `json:"image_path"`
	TopKJSON   string    `json:"top_k_json"`
	CreatedAt  time.Time `json:"created_at"`
}
