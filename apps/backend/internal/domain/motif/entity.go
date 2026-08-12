package motif

import "time"

type Motif struct {
	ID          uint      `json:"id" gorm:"primaryKey"`
	Name        string    `json:"name" gorm:"uniqueIndex"`
	Origin      string    `json:"origin"`
	Description string    `json:"description"`
	Philosophy  string    `json:"philosophy"`
	ImageURL    string    `json:"image_url"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}
