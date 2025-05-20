import mlflow
from mlflow.tracking import MlflowClient

def promote_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    
    # Находим лучший run по метрике accuracy
    runs = client.search_runs(
        experiment_ids=["1"],
        filter_string="metrics.accuracy > 0.9",
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    if not runs:
        print("No suitable model found")
        exit(1)
    
    best_run = runs[0]
    
    # Регистрируем модель
    model_uri = f"runs:/{best_run.info.run_id}/model"
    client.create_registered_model(
        name="IrisClassifier"
    )
    mv = client.create_model_version(
        name="IrisClassifier",
        source=model_uri,
        run_id=best_run.info.run_id
    )
    
    # Переводим модель в "Staging"
    client.transition_model_version_stage(
        name="IrisClassifier",
        version=mv.version,
        stage="Staging"
    )

if __name__ == "__main__":
    promote_model()