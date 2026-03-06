# 01_churn_data_simulation.py
# Objective: Generate synthetic policyholder data to demonstrate Churn Prediction (Classification).
# We embed hidden logic where high support calls and payment delays lead to high churn probability.

from pyspark import SparkSession
from pyspark.sql.functions import col, when, rand, randn, round

spark = SparkSession.builder.appName("Concentrix_Churn_Simulation").getOrCreate()
print("Starting Policyholder Data Simulation for Churn Engine...")

# 1. Generate Base DataFrame (100,000 policyholders)
num_customers = 100000
df_base = spark.range(0, num_customers).withColumnRenamed("id", "customer_id")

# 2. Engineer Behavioral and Demographic Features
print("Engineering features (Tenure, Premiums, Support Calls, Payment Delays)...")

df_features = df_base \
    .withColumn("tenure_months", round(1 + rand() * 119)) \
    .withColumn("monthly_premium", round(50 + (randn() * 20) + (rand() * 100), 2)) \
    .withColumn("support_calls_last_6m", 
                when(rand() < 0.7, 0) # 70% of people don't call support at all
                .otherwise(round(1 + (rand() * 5)))) \
    .withColumn("payment_delays_days", 
                 when(rand() < 0.8, 0) # 80% pay perfectly on time
                 .otherwise(round(1 + (rand() * 20)))) \
    .withColumn("claims_filed", 
                when(rand() < 0.85, 0) # 85% haven't filed a claim recently
                .otherwise(round(1 + (rand() * 2))))
                
# 3. Calculate Hidden "Churn Risk" Probability (The Answer Key)
# High support calls + low tenure + payment delays = extremely high churn risk
df_risk = df_features .withColumn("churn_probability",
    (col("support_calls_last_6m") * 0.15) + 
    (col("payment_delays_days") * 0.02) + 
    (when(col("tenure_months") < 12, 0.2).otherwise(0)) - 
    (col("tenure_months") * 0.001) + 
    (rand() * 0.2) # Add some random noise so it isn't too easy for the ML model
)

# 4. Assign the actual target label (1 = Churned, 0 = Retained)
df_final = df_risk \
    .withColumn("churned", when (col("churn_probability") > 0.6, 1).otherwise(0)) \
    .drop("churn_probability") # Drop the risk column so it's not directly visible to the model

# Clean up any negative premiums caused by the random normal distribution
df_final = df_final.withColumn("monthly_premium", when(col("monthly_premium") < 20, 20.0).otherwise(col("monthly_premium")))

# 5. Save to Unity Catalog Managed Delta Table
output_table = "workspace.default.concentrix_simulated_churn"
print(f"Saving data to Unity Catalog: {output_table}...")

df_final.write.format("delta").mode("overwrite").saveAsTable(output_table)
print("✅ Data simulation complete! Ready for Classification Feature Engineering.")
display(df_final)