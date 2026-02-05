Perfect — **Section 2 (Data Processing)** is very hands-on and **heavily tested** in the Databricks ML Associate exam.
Below is a **complete, exam-grade study guide** with **definitions, why/when/how, Spark code, pros/cons, and traps**.

You can use this as **final revision notes**.

---

# 📘 Section 2 – Data Processing (Databricks ML Associate)

> **Exam goal:**
> Test whether you can **explore, clean, transform, and prepare data using Spark**, and whether you understand **why one technique is chosen over another**.

---

## 1️⃣ Compute Summary Statistics on a Spark DataFrame

---

## 🔹 Using `.summary()`

### **What**

Computes descriptive statistics for **numeric columns**.

### **How**

```python
df.summary().show()
```

### **What it returns**

* count
* mean
* stddev
* min
* max
* percentiles (25%, 50%, 75%)

### **Why use it**

* Quick EDA
* Detect skew, outliers, scale

### **Pros**

✅ Fast
✅ Built-in
✅ No config

### **Cons**

❌ Numeric columns only
❌ Limited customization

📌 **Exam rule**

> `.summary()` = numeric EDA

---

## 🔹 Using `dbutils.data.summarize`

### **What**

Databricks UI-driven data summary.

```python
dbutils.data.summarize(df)
```

### **Why**

* Visual profiling
* Column-level insights

📌 **Exam**

> `dbutils.data.summarize` is **exploratory**, not for pipelines.

---

## 2️⃣ Remove Outliers from a Spark DataFrame

---

## 🔹 Method 1: Standard Deviation (Z-score)

### **Definition**

Remove values far from the mean.

### **How**

```python
from pyspark.sql.functions import col, mean, stddev

stats = df.select(
    mean("value").alias("mean"),
    stddev("value").alias("std")
).collect()[0]

filtered_df = df.filter(
    (col("value") >= stats.mean - 3 * stats.std) &
    (col("value") <= stats.mean + 3 * stats.std)
)
```

### **When to use**

✔ Normally distributed data

### **Pros**

✅ Simple
✅ Works well for Gaussian data

### **Cons**

❌ Sensitive to extreme outliers
❌ Poor for skewed data

---

## 🔹 Method 2: IQR (Interquartile Range)

### **Definition**

Uses percentiles instead of mean.

### **How**

```python
q1, q3 = df.approxQuantile("value", [0.25, 0.75], 0.0)
iqr = q3 - q1

filtered_df = df.filter(
    (col("value") >= q1 - 1.5 * iqr) &
    (col("value") <= q3 + 1.5 * iqr)
)
```

### **When to use**

✔ Skewed data
✔ Non-normal distributions

### **Pros**

✅ Robust
✅ Less sensitive to extreme values

### **Cons**

❌ Slightly more complex

📌 **Exam rule**

> Skewed data → **IQR**

---

## 3️⃣ Create Visualizations for Features

---

## 🔹 Categorical Features

### **Best plots**

* Bar chart
* Count plot

### **Example**

```python
df.groupBy("category").count().display()
```

📌 **Exam**

> Bar charts for categorical data

---

## 🔹 Continuous Features

### **Best plots**

* Histogram
* Box plot

```python
df.select("value").display()
```

📌 **Exam**

> Histograms for continuous data

---

## 4️⃣ Compare Features (Categorical vs Continuous)

---

### 🔹 Two Continuous Features

| Method       | Use Case            |
| ------------ | ------------------- |
| Correlation  | Linear relationship |
| Scatter plot | Visual relationship |

```python
df.stat.corr("x", "y")
```

---

### 🔹 Two Categorical Features

| Method            | Use Case     |
| ----------------- | ------------ |
| Contingency table | Relationship |
| Chi-square test   | Independence |

📌 **Exam trap**
❌ Correlation ≠ categorical data

---

## 5️⃣ Imputing Missing Values – Mean vs Median vs Mode

---

## 🔹 Definitions

| Method | Definition    |
| ------ | ------------- |
| Mean   | Average       |
| Median | Middle value  |
| Mode   | Most frequent |

---

## 🔹 Comparison (VERY IMPORTANT)

| Method | Best When      | Avoid When       |
| ------ | -------------- | ---------------- |
| Mean   | Symmetric data | Outliers present |
| Median | Skewed data    | Small samples    |
| Mode   | Categorical    | Continuous data  |

📌 **Exam rule**

> Outliers → **median**

---

## 6️⃣ Impute Missing Values in Spark

---

### 🔹 Mean / Median

```python
from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=["age"],
    outputCols=["age_imputed"],
    strategy="median"
)

df_imputed = imputer.fit(df).transform(df)
```

---

### 🔹 Mode (categorical)

```python
mode = df.groupBy("category").count().orderBy("count", ascending=False).first()[0]

df_imputed = df.fillna({"category": mode})
```

---

## 7️⃣ One-Hot Encoding

---

## 🔹 What is One-Hot Encoding?

Converts categories into binary columns.

```
Color = Red, Blue
→ Red=[1,0], Blue=[0,1]
```

---

## 🔹 How in Spark

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder

indexer = StringIndexer(inputCol="color", outputCol="color_idx")
encoder = OneHotEncoder(inputCol="color_idx", outputCol="color_vec")

df = indexer.fit(df).transform(df)
df = encoder.fit(df).transform(df)
```

---

## 8️⃣ When One-Hot Encoding IS / IS NOT Appropriate

---

### 🔹 Appropriate for

✔ Linear models
✔ Logistic regression
✔ Small cardinality categories

---

### 🔹 NOT appropriate for

❌ Tree-based models
❌ High-cardinality categories

📌 **Exam rule**

> Tree models do **not** require one-hot encoding.

---

## 9️⃣ Log Scale Transformation

---

## 🔹 What

Apply log to reduce skew.

```python
from pyspark.sql.functions import log

df = df.withColumn("log_sales", log(col("sales")))
```

---

## 🔹 When to use

✔ Right-skewed data
✔ Large numeric ranges
✔ Multiplicative relationships

---

## 🔹 When NOT to use

❌ Zero or negative values
❌ Already normalized data

📌 **Exam**

> Log transform stabilizes variance

---

## 🧠 COMMON EXAM TRAPS (SECTION 2)

❌ Using mean with outliers
❌ One-hot encoding tree models
❌ Using correlation for categorical data
❌ Forgetting Spark ML transformers

---

## 🧠 FINAL MEMORY BLOCK (SECTION 2)

```
.summary() → stats
IQR → skewed outliers
Mean vs Median → symmetry
One-hot → linear models
Log → skew reduction
```

---

## 🎯 What the exam REALLY tests in Section 2

✔ Correct method selection
✔ Understanding *why*, not just *how*
✔ Spark-native processing
✔ Avoiding common ML mistakes

---

### Next steps (your choice):

* **Section 3 – Model Development (algorithms, tuning, metrics)**
* **Rapid MCQs for Section 2**
* **End-to-end EDA → Feature Engineering example**

Just tell me 👍
