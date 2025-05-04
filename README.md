# Big Data Coursework: Distributed Data Processing and Cloud-Based Machine Learning

## Overview

This project showcases end-to-end experimentation with large-scale parallel processing of image data using **Apache Spark**, **TensorFlow**, and **Google Cloud Dataproc**. It focuses on cloud-based distributed data pre-processing, performance benchmarking, and architectural insights, anchored in the practical analysis of the publicly available **Flowers dataset** (3,600 images, 5 classes).

The coursework is aligned with the **INM432 Big Data module**, leveraging modern data science tooling and scalable infrastructure to explore both implementation and theoretical aspects of cloud-based data processing systems.

---

## Objectives

- Implement **parallelized image preprocessing** with both `tf.data` (TensorFlow) and Spark RDDs.
- Measure and analyze **I/O throughput** in different Google Cloud configurations.
- Compare speed-ups from **TFRecord-based** storage vs. raw image decoding.
- Evaluate the performance impact of **cluster size, disk types, and resource allocation**.
- Discuss theoretical implications based on academic literature (`Cherrypick` paper).

---

## Technologies Used

- **TensorFlow 2.15** (`tf.data`, `TFRecord`, `decode_jpeg`)
- **Apache Spark 3.5.0**
- **Google Cloud Platform**
  - Dataproc clusters
  - Google Cloud Storage (GCS)
- **Colab for development/testing**
- **Scikit-learn & Matplotlib** (for regression and plotting)
- Python libraries: `pyspark`, `numpy`, `tensorflow`, `pickle`, `matplotlib`, `sklearn`

---

## Project Structure

### 📁 Section 0: Setup and Authentication
- Environment setup for TensorFlow and Spark.
- Authentication to Google Cloud Storage and Dataproc.
- Bucket creation and directory setup.

### 📁 Section 1: Image Preprocessing
#### ✅ Task 1a: TFRecord Generation in Spark
- Converted `tf.data` pipelines (decode → resize → crop → recompress) into Spark RDD transformations.
- Used `mapPartitionsWithIndex()` to write `TFRecord` files in parallel.

#### ✅ Task 1b: Verification
- Read and visualized images from generated TFRecords.
- Successfully validated integrity using custom display functions.

#### ✅ Task 1c: Single-Node Cluster Deployment
- Deployed Dataproc cluster with max SSD and vCPU on a single VM.
- Ran preprocessing job in cloud and verified output.

#### ✅ Task 1d: Cluster Optimisation
- Ran jobs on:
  - 1 master + 7 workers (1 vCPU)
  - 4 VMs (2 vCPUs)
  - Single high-resource node (16 vCPUs)
- Measured runtime and cloud monitoring stats (CPU/network utilization).
- Observed **better utilization** with `RDD.parallelize(..., numPartitions=16)`.

---

### 📁 Section 2: Speed Tests & Benchmarking
#### ✅ Task 2a: Spark-Parallel Speed Tests
- Designed `process_dataset(batch_size, num_batches)` logic.
- Tested combinations in parallel on Spark RDDs.
- Measured **images/sec** throughput using `TFRecordDataset`.

#### ✅ Task 2b: Output Collection & Analysis
- Collected results into `average_performance_TIMESTAMP.pkl`.
- Created regression models to understand how `batch_size`, `num_batches`, and their product impact throughput.
- Produced clean summary tables and visualizations.

#### ✅ Task 2c: Optimisation
- Used `.cache()` on Spark RDDs to **minimize recomputation** during aggregations.
- Observed notable reduction in runtime for repeated computations.

#### ✅ Task 2d: Modelling & Interpretation
- Used `LinearRegression` to fit throughput as a function of parameters.
- Identified that **batch_product** is the strongest predictor of read speed.
- Interpreted slope coefficients and intercept values.
- Discussed scaling implications, cloud read limits, and I/O bottlenecks.

---

### 📁 Section 3: Theoretical Discussion
#### ✅ Task 3a: Cherrypick Contextualisation
- Evaluated how adaptive cloud configuration prediction aligns with findings.
- Highlighted variability in performance based on hidden factors (disk speed, scheduling delays).

#### ✅ Task 3b: Strategic Recommendations
- For **batch workloads**: Prefer TFRecord format, distribute I/O using more disks over more nodes.
- For **streaming**: Consider lower batch sizes and finer control of task distribution.
- Emphasized importance of **profiling and dynamic tuning** in cloud jobs.

---

## Key Learnings

- 📌 **Data Format Matters**: Using `TFRecord` files yields dramatic improvements in read speed vs. raw JPEGs.
- 📌 **Spark + TensorFlow**: Achieving distributed pre-processing requires careful reshaping of TensorFlow operations into stateless Spark RDD transformations.
- 📌 **Cloud Infrastructure Strategy**: Small clusters with high I/O (standard disk) may underperform compared to larger clusters with better distributed disks and CPUs.
- 📌 **Profiling is Essential**: Google's monitoring dashboards were crucial for identifying CPU/network bottlenecks.
- 📌 **Parallel Speed Tests Are Non-Trivial**: Spark-based performance benchmarking helped simulate production-scale load testing.

---

## Final Thoughts

This project successfully bridges practical machine learning operations with systems-level thinking. It highlights how cloud-native data pipelines require not only correct implementation but also **strategic infrastructure planning and optimisation**.

The experience has deepened my understanding of:
- Cloud-based parallelism
- TFRecord pipelines
- Spark's partitioning logic
- Measuring system throughput
- Applying regression models to system metrics

---

## How to Run

1. Set up a GCP project, enable billing and Dataproc.
2. Create buckets and adjust `PROJECT` and `BUCKET` constants.
3. Clone and run the notebook locally in Colab for testing.
4. Use `gcloud` CLI to deploy clusters and submit Spark jobs.
