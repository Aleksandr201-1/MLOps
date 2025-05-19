import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Загрузка данных
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

# Настройка MLFlow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Iris_Classification")

with mlflow.start_run():
    # Параметры модели
    params = {"n_estimators": 100, "max_depth": 5}
    
    # Обучение модели
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Логирование
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "random_forest_model")