variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "europe-west3"
}

variable "api_image" {
  description = "Full Artifact Registry path + tag for the API container"
  type        = string
}

variable "demo_image" {
  description = "Full Artifact Registry path + tag for the demo container"
  type        = string
}
