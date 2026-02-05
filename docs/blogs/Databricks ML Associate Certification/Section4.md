Perfect.
Below is **Section 4 – Model Deployment** written as a **complete study resource for the Databricks Machine Learning Associate**, from the perspective of a **Databricks ML practitioner** (not Azure ML, not generic ML).

This is **exam-oriented**, but also **practical**, with:

* clear definitions
* **Databricks-native code patterns**
* when / why / how
* pros & cons
* common exam traps

You can copy this directly into your notes.

---

# 📘 Section 4 – Model Deployment (Databricks ML Associate)

> **Big picture (EXAM VIEW):**
> Databricks tests whether you understand **how models are deployed and consumed on Databricks**, not cloud infra.

Databricks supports **three deployment styles**:

1. **Batch inference**
2. **Real-time inference (Model Serving)**
3. **Streaming inference (Delta Live Tables)**

---

## 1️⃣ Model Serving Approaches: Batch vs Realtime vs Streaming

---

## 🔹 Batch Inference

### **Definition**

Batch inference means:

> Running predictions on a **large dataset at once**, usually offline.

Typical inputs:

* Delta tables
* Parquet files
* Pandas DataFrames

Typical outputs:

* Delta tables
* Files

---

### **How it works in Databricks**

* Load a trained model
* Apply it to a dataset
* Save predictions

Usually run as:

* Notebook job
* Scheduled workflow

---

### **Example (Pandas batch inference)**

```python
import mlflow
import pandas as pd

model_uri = "models:/sales_model/Production"
model = mlflow.pyfunc.load_model(model_uri)

df = pd.read_csv("/dbfs/data/sales.csv")
predictions = model.predict(df)

df["prediction"] = predictions
df
```

---

### **Why use batch inference**

* Large volumes of data
* No strict latency requirements
* Cost-efficient

---

### **Pros**

✅ Scales well
✅ Simple to implement
✅ Cheap

### **Cons**

❌ Not real-time
❌ Results delayed

---

### **When to use (EXAM)**

✔ Daily forecasts
✔ Offline scoring
✔ Periodic reports

---

## 🔹 Real-time Inference (Model Serving)

### **Definition**

Real-time inference means:

> Serving predictions **on demand** via an HTTP endpoint.

Databricks uses **Model Serving Endpoints**.

---

### **How it works**

* Register a model in MLflow
* Deploy it to a **serving endpoint**
* Clients send requests → receive predictions instantly

---

### **Key Databricks Concept**

> **Model Serving is managed by Databricks**
> (no AKS, no Kubernetes config needed)

---

### **Deploy a model to a serving endpoint**

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

w.serving_endpoints.create(
    name="sales-realtime-endpoint",
    config={
        "served_models": [{
            "model_name": "sales_model",
            "model_version": "1",
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]
    }
)
```

---

### **Query a real-time endpoint**

```python
import requests
import json

response = requests.post(
    "https://<databricks-url>/serving-endpoints/sales-realtime-endpoint/invocations",
    headers={
        "Authorization": "Bearer <TOKEN>",
        "Content-Type": "application/json"
    },
    data=json.dumps({
        "dataframe_records": [{"feature1": 10, "feature2": 20}]
    })
)

print(response.json())
```

---

### **Why use real-time inference**

* Low-latency predictions
* Interactive applications
* APIs, dashboards

---

### **Pros**

✅ Low latency
✅ Scalable
✅ Fully managed

### **Cons**

❌ More expensive than batch
❌ Not ideal for huge datasets

---

### **When to use (EXAM)**

✔ APIs
✔ User-facing apps
✔ On-demand scoring

---

## 🔹 Streaming Inference (Delta Live Tables)

---

### **Definition**

Streaming inference means:

> Applying a model **continuously** to streaming data.

Databricks uses:

* **Delta Live Tables (DLT)**
* Structured Streaming

---

### **How it works**

* Data arrives as a stream
* Model is applied to each micro-batch
* Predictions are written continuously

---

### **Streaming inference with DLT (KEY CONCEPT)**

```python
import mlflow
from pyspark.sql.functions import struct
import dlt

model_uri = "models:/sales_model/Production"
model = mlflow.pyfunc.spark_udf(spark, model_uri)

@dlt.table
def streaming_predictions():
    df = spark.readStream.table("incoming_sales")
    return df.withColumn(
        "prediction",
        model(struct(*df.columns))
    )
```

---

### **Why use streaming inference**

* Real-time data streams
* Continuous predictions
* Event-driven systems

---

### **Pros**

✅ Real-time at scale
✅ Fault tolerant
✅ Declarative pipelines

### **Cons**

❌ More complex
❌ Requires streaming data

---

### **When to use (EXAM)**

✔ IoT
✔ Event streams
✔ Continuous scoring

---

## 🔁 Comparison Table (MEMORIZE)

| Feature   | Batch              | Realtime      | Streaming      |
| --------- | ------------------ | ------------- | -------------- |
| Latency   | High               | Low           | Near real-time |
| Data size | Large              | Small         | Continuous     |
| Trigger   | Manual / scheduled | API call      | Stream         |
| Cost      | Low                | Medium        | Medium         |
| Tool      | Pandas / Spark     | Model Serving | DLT            |

---

## 2️⃣ Deploy a Custom Model to a Model Endpoint

---

### **What is a “custom model”?**

Any MLflow model:

* sklearn
* XGBoost
* PyTorch
* Custom `pyfunc`

---

### **Register model first**

```python
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    registered_model_name="sales_model"
)
```

Then deploy to:

* Serving endpoint (real-time)
* Notebook/job (batch)
* DLT pipeline (streaming)

---

## 3️⃣ Use Pandas to Perform Batch Inference

---

### **Why Pandas is tested**

Databricks ML Associate assumes:

* You know **simple batch inference**
* Not Spark-only

---

### **Pattern to recognize (EXAM)**

```python
model = mlflow.pyfunc.load_model(model_uri)
preds = model.predict(pandas_df)
```

📌 **Key point**

> Pandas batch inference runs on the **driver**, not distributed.

---

## 4️⃣ Streaming Inference with Delta Live Tables

---

### **Key idea (EXAM GOLD)**

> **DLT is how Databricks does production-grade streaming inference**

---

### **Why DLT**

* Automatic retries
* Monitoring
* Lineage
* Schema enforcement

---

### **Exam trap**

❌ Streaming inference ≠ model serving
✔ Streaming inference = **DLT + streaming source**

---

## 5️⃣ Split Traffic Between Endpoints (Realtime Inference)

---

### **Definition**

Traffic splitting allows:

> Sending a percentage of requests to different model versions.

Used for:

* A/B testing
* Canary deployments

---

### **How it works**

One endpoint → multiple served models

```json
"served_models": [
  {"model_version": "1", "traffic_percentage": 90},
  {"model_version": "2", "traffic_percentage": 10}
]
```

---

### **Why this matters**

* Safe rollouts
* Compare performance
* Reduce risk

---

### **Pros**

✅ Controlled deployment
✅ Easy rollback

### **Cons**

❌ Slightly more complexity

---

### **When to use (EXAM)**

✔ Model version comparison
✔ Gradual rollout

---

## 🧠 COMMON EXAM TRAPS (VERY IMPORTANT)

❌ Confusing **batch inference** with **streaming inference**
❌ Thinking model serving is Kubernetes-based
❌ Using Spark for real-time APIs
❌ Thinking Pandas batch inference is distributed

---

## 🧠 FINAL MEMORY BLOCK (LOCK THIS IN)

```
Batch → Pandas / Spark
Realtime → Model Serving Endpoint
Streaming → Delta Live Tables
Split traffic → A/B testing
```

---

## 🎯 What the exam REALLY tests in Section 4

✔ Correct deployment type selection
✔ Knowing Databricks-native tools
✔ Understanding trade-offs
✔ Recognizing MLflow deployment patterns

---

### Next steps (you choose):

* **Section 3 – Model Development (algorithms, tuning, metrics)**
* **Rapid-fire MCQs for Section 4**
* **End-to-end MLflow → Serving workflow walkthrough**

Just tell me 👍
