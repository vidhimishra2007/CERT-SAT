from backend_telemetry_model import train_global_model


CSV_PATH = "data/backend_telemetry_history.csv"
OUTPUT_PATH = "models_saved/backend_telemetry_isoforest.joblib"


def main():
    artifact = train_global_model(CSV_PATH, OUTPUT_PATH)
    print(f"Saved model: {OUTPUT_PATH}")
    print(f"Raw input features: {artifact['raw_feature_count']}")
    print(f"Model input features: {artifact['model_feature_count']}")
    print(f"Window size: {artifact['window_size']}")


if __name__ == "__main__":
    main()
