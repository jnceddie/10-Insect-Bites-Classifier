import requests
import json

# Adjust this URL if your Flask app runs on a different port
API_URL = "http://127.0.0.1:5000/visualize_pipeline"

def test_visualize_pipeline():
    response = requests.get(API_URL)
    assert response.status_code == 200, f"Status code: {response.status_code}"
    data = response.json()
    assert data.get('success'), f"API did not return success: {data}"
    pipeline_data = data.get('pipeline_data')
    assert pipeline_data is not None, "No pipeline_data returned"
    steps = pipeline_data.get('pipeline_steps')
    assert steps is not None and isinstance(steps, list), "pipeline_steps missing or not a list"
    assert len(steps) >= 12, f"Expected at least 12 steps, got {len(steps)}"
    for idx, step in enumerate(steps):
        print(f"Step {idx+1}: {step.get('title')}")
    print("All pipeline steps are present and accessible.")

if __name__ == "__main__":
    test_visualize_pipeline()
