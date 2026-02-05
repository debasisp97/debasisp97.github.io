Perfect capstone question 👍
Below is a **complete, end-to-end PySpark ML pipeline** written **exactly in the Databricks ML Associate + Unity Catalog + MLflow way**.

This is **production-grade**, **exam-aligned**, and reusable.

I’ll cover **everything in one flow**:

1. Read data (numeric + categorical)
2. Feature engineering pipeline
3. Train/test split
4. Train a model
5. Track with MLflow
6. Register model in **Unity Catalog**
7. Create & register **Feature Store table in UC**
8. Reuse features for **retraining**
9. Reuse model + features for **inference**

You can treat this as **the final mental model for the exam**.

---

# 0️⃣ Assumptions (Context)

* Spark DataFrame input
* Binary classification example
* Unity Catalog enabled
* Databricks ML Runtime
* Table names are UC-qualified

---

# 1️⃣ Read Data (Numeric + Categorical)

```python
df = spark.read.table("main.raw.customer_data")

df.printSchema()
```

Example columns:

* `customer_id` (key)
* `age` (numeric)
* `income` (numeric)
* `country` (categorical)
* `churn` (label)

---

# 2️⃣ Feature Engineering Pipeline (Spark ML)

## Separate feature types

```python
numeric_cols = ["age", "income"]
categorical_cols = ["country"]
label_col = "churn"
```

---

## Transformers

```python
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer
)
```

### Impute numeric values

```python
imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[f"{c}_imputed" for c in numeric_cols],
    strategy="median"
)
```

### Encode categorical values

```python
indexer = StringIndexer(
    inputCol="country",
    outputCol="country_idx",
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCol="country_idx",
    outputCol="country_vec"
)
```

### Assemble features

```python
assembler = VectorAssembler(
    inputCols=["age_imputed", "income_imputed", "country_vec"],
    outputCol="features"
)
```

---

# 3️⃣ Model (Estimator)

```python
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features",
    labelCol=label_col
)
```

---

# 4️⃣ Full Pipeline

```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[
    imputer,
    indexer,
    encoder,
    assembler,
    lr
])
```

📌 **Exam key**

> Estimators + transformers chained → Pipeline

---

# 5️⃣ Train / Test Split

```python
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
```

---

# 6️⃣ Train + Track with MLflow

```python
import mlflow
import mlflow.spark

mlflow.set_experiment("/Shared/churn_experiment")

with mlflow.start_run():
    model = pipeline.fit(train_df)

    preds = model.transform(test_df)

    mlflow.spark.log_model(
        spark_model=model,
        artifact_path="model",
        registered_model_name="main.ml_models.churn_model"
    )

    mlflow.log_param("model_type", "LogisticRegression")
```

📌 **Exam**

* Spark ML model → `mlflow.spark.log_model`
* UC model name is **3-level namespace**

---

# 7️⃣ Create Feature Store Table (Unity Catalog)

## Create features DataFrame

```python
features_df = df.select(
    "customer_id",
    "age",
    "income",
    "country"
)
```

---

## Register Feature Store table

```python
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

fs.create_table(
    name="main.ml_features.customer_features",
    primary_keys=["customer_id"],
    df=features_df,
    description="Customer features for churn prediction"
)
```

---

## Write / Update features

```python
fs.write_table(
    name="main.ml_features.customer_features",
    df=features_df,
    mode="merge"
)
```

📌 **Exam**

> Feature Store tables are **Delta tables + metadata**

---

# 8️⃣ Train Using Feature Store (Reusable Training)

```python
from databricks.feature_store import FeatureLookup

lookups = [
    FeatureLookup(
        table_name="main.ml_features.customer_features",
        lookup_key="customer_id"
    )
]

training_set = fs.create_training_set(
    df=df.select("customer_id", label_col),
    feature_lookups=lookups,
    label=label_col
)

training_df = training_set.load_df()
```

Now reuse **same pipeline**:

```python
model = pipeline.fit(training_df)
```

📌 **Exam**

> Feature Store prevents training-serving skew

---

# 9️⃣ Re-Training with Same Features (MLOps Pattern)

* Feature logic unchanged
* New data arrives
* Just rerun training pipeline
* Log new run → register new version

✔ **Model promotion, not code promotion**

---

# 🔟 Batch Inference Using Feature Store

```python
scoring_df = spark.read.table("main.raw.new_customers")

scoring_set = fs.create_training_set(
    df=scoring_df.select("customer_id"),
    feature_lookups=lookups
)

scoring_features = scoring_set.load_df()

predictions = model.transform(scoring_features)
```

---

# 1️⃣1️⃣ Load Registered Model for Inference

```python
import mlflow.spark

model_uri = "models:/main.ml_models.churn_model@Champion"
model = mlflow.spark.load_model(model_uri)

preds = model.transform(scoring_features)
```

📌 **Exam**

> Use aliases (`Champion`) instead of versions

---

# 1️⃣2️⃣ Promote Model Using Alias (Champion / Challenger)

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    name="main.ml_models.churn_model",
    alias="Champion",
    version="2"
)
```

---

# 1️⃣3️⃣ What You Have Achieved (EXAM SUMMARY)

✔ Spark ML pipeline
✔ Numeric + categorical processing
✔ One-hot encoding (correct usage)
✔ Feature Store in Unity Catalog
✔ MLflow tracking
✔ UC model registry
✔ Reusable training
✔ Reusable inference
✔ Champion / challenger MLOps

---

# 🧠 FINAL MEMORY BLOCK (LOCK THIS IN)

```
Spark DF → Pipeline
Pipeline → MLflow
Features → Feature Store (UC)
Model → UC Registry
Reuse → No skew
Alias → Champion
```

---

## 🎯 This single example covers ~40–50% of the exam concepts

If you want next, I can:

* Convert this into **MCQs**
* Show **AutoML → Feature Store → Serving**
* Do a **full mock exam**
* Or give you a **1-page final revision sheet**

Just tell me 👍
