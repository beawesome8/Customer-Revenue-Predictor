terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "apis" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
  ])
  service            = each.key
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "purchase_intent_gcp" {
  location      = var.region
  repository_id = "purchase-intent-gcp"
  format        = "DOCKER"
  description   = "Purchase intent XGBoost API and demo images"
  depends_on    = [google_project_service.apis]
}

resource "google_cloud_run_v2_service" "api" {
  name     = "purchase-intent-api"
  location = var.region

  template {
    containers {
      image = var.api_image
      resources {
        limits            = { memory = "512Mi", cpu = "1000m" }
        cpu_idle          = true
        startup_cpu_boost = true
      }
    }
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "api_public" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service" "demo" {
  name     = "purchase-intent-demo"
  location = var.region

  template {
    containers {
      image = var.demo_image
      resources {
        limits            = { memory = "512Mi", cpu = "1000m" }
        cpu_idle          = true
        startup_cpu_boost = true
      }
      env {
        name  = "API_URL"
        value = google_cloud_run_v2_service.api.uri
      }
    }
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "demo_public" {
  location = google_cloud_run_v2_service.demo.location
  name     = google_cloud_run_v2_service.demo.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
