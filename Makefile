# ============================================================
# Sentinel Fraud Detection — Makefile
# ============================================================

.PHONY: install featurize train-lgbm train-xgb train-rf train-catboost \
        train-all ensemble evaluate dashboard test dvc-repro docker clean

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	dvc init --no-scm 2>/dev/null || true
	@echo "✅ Dependencies installed."

# ─── DVC Pipeline ─────────────────────────────────────────────────────────────
dvc-repro:
	dvc repro
	@echo "✅ DVC pipeline reproduced."

dvc-status:
	dvc status

# ─── Feature Store & Engineering ───────────────────────────────────────────────
feast-prepare:
	python feature_store/prepare_feature_store.py
	@echo "✅ Feast online and offline feature stores prepared."

featurize:
	python src/features/run_feature_engineering.py
	@echo "✅ Feature engineering complete."

# ─── Training ─────────────────────────────────────────────────────────────────
train-lgbm:
	python training/train.py --model lgbm
	@echo "✅ LightGBM trained."

train-lgbm-opt:
	python training/train.py --model lgbm --optimize
	@echo "✅ LightGBM tuned with Optuna."

train-xgb:
	python training/train.py --model xgb
	@echo "✅ XGBoost trained."

train-xgb-opt:
	python training/train.py --model xgb --optimize
	@echo "✅ XGBoost tuned with Optuna."

train-rf:
	python training/train.py --model rf
	@echo "✅ Random Forest trained."

train-catboost:
	python training/train.py --model catboost
	@echo "✅ CatBoost trained."

train-catboost-opt:
	python training/train.py --model catboost --optimize
	@echo "✅ CatBoost tuned with Optuna."

train-all: train-lgbm train-xgb train-rf train-catboost
	@echo "✅ All individual models trained."

ensemble:
	python training/train_ensemble.py
	@echo "✅ Ensemble pipeline built."

evaluate:
	python training/evaluate_threshold.py
	@echo "✅ Threshold evaluation complete."

# ─── Dashboard & API ──────────────────────────────────────────────────────────
dashboard:
	streamlit run app/main.py
	@echo "🚀 Sentinel dashboard started at http://localhost:8501"

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
	@echo "🚀 Sentinel REST API running at http://localhost:8000 (Docs: /docs)"

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "✅ All tests passed."

test-fast:
	pytest tests/ -v -x
	@echo "✅ Fast test run complete."

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker compose -f docker/docker-compose.yml build
	@echo "✅ Docker image built."

docker-up:
	docker compose -f docker/docker-compose.yml up
	@echo "🚀 Sentinel running in Docker at http://localhost:8501"

docker-down:
	docker compose -f docker/docker-compose.yml down

# ─── Utilities ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cache cleaned."

clean-db:
	rm -f data/alerts.db
	@echo "✅ Alert database cleared."

clean-models:
	rm -f models/ensemble_pipeline.pkl models/ordinal_encoder.pkl models/feature_cols.json
	@echo "✅ Trained ensemble cleared (individual models preserved)."

logs:
	mkdir -p logs
	@echo "✅ Logs directory ready."

metrics:
	mkdir -p metrics
	@echo "✅ Metrics directory ready."
