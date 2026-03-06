# 02_feature_engineering.py
# Objective: Prepare the simulated churn data for tree-based classification.
# We will assemble the feature vector and perform an 80/20 Train/Test split.

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("Concentrix_Feature_Prep").getOrCreate()
print("Starting Feature Engineering for Churn Classification...")

# 1. Load the raw simulated data
input_table = "workspace.default.concentrix_simulated_churn"
df_raw = spark.table(input_table)

# 2. Define the input features 
feature_cols = [
    "tenure_months", 
    "monthly_premium", 
    "support_calls_last_6m", 
    "payment_delays_days", 
    "claims_filed"
] 

# 3. Assemble the features into a single vector
# Note: We are deliberately NOT using StandardScaler here because we are building
# an XGBoost/Gradient Boosted Tree model, which is scale-invariant.
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df_raw)



# 3. Assemble the features into a single vector
# Note: We are deliberately NOT using StandardScaler here because we are building
# an XGBoost/Gradient Boosted Tree model, which is scale-invariant.
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_assembled = assembler.transform(df_raw)

# Keep only the columns we need for ML: lead_id, features, and the target label (churned)
modeling_dataset = df_assembled.select("customer_id", "features", "churned")

# 4. The 80/20 Train/Test Split
# We use a seed (42) so the random split is reproducible every time we run the script
train_data, test_data = modeling_dataset.randomSplit([0.8, 0.2], seed=42)

train_count = train_data.count()
test_count = test_data.count()
print(f"Training Set: {train_count} records")
print(f"Testing Set: {test_count} records")

# 5. Save the Train and Test sets back to Unity Catalog
train_table = "workspace.default.concentrix_churn_train"
test_table = "workspace.default.concentrix_churn_test"

print(f"Saving training data to Unity Catalog: {train_table}...")
train_data.write.mode("overwrite").saveAsTable(train_table)

print(f"Saving testing data to Unity Catalog: {test_table}...")
test_data.write.mode("overwrite").saveAsTable(test_table)

print("✅ Feature Engineering & Data Splitting complete! Ready for ML Modeling.")
display(train_data.limit(5))