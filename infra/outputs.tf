output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "demo_url" {
  value = google_cloud_run_v2_service.demo.uri
}
