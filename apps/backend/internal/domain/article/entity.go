package article

import "time"

type Article struct {
	ID           uint      `json:"id" gorm:"primaryKey"`
	Title        string    `json:"title"`
	Slug         string    `json:"slug" gorm:"uniqueIndex"`
	Content      string    `json:"content"`
	ThumbnailURL string    `json:"thumbnail_url"`
	CategoryID   uint      `json:"category_id"`
	AuthorID     uint      `json:"author_id"`
	CreatedAt    time.Time `json:"created_at"`
	UpdatedAt    time.Time `json:"updated_at"`
}
