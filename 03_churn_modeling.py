# 03_churn_modeling.py
# Objective: Train a Gradient Boosted Tree (GBT) classification model to predict
# customer churn. Evaluate using AUC-ROC and extract Feature Importances.

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Concentrix_Churn_Modeling").getOrCreate()
print("Starting Churn Classification Modeling Phase...")

# 1. Load the Train and Test sets
train_table = "workspace.default.concentrix_churn_train"
test_table = "workspace.default.concentrix_churn_test"

train_data = spark.table(train_table)
test_data = spark.table(test_table)

# 2. Initialize the Gradient Boosted Tree (GBT) Classifier
# GBTs build trees sequentially, where each new tree tries to correct the errors of the previous one.
print("Initializing the GBT Classifier...")
gbt = GBTClassifier(
    labelCol="churned", 
    featuresCol="features", 
    maxIter=20, # Number of trees in the forest
    maxDepth=5, # Maximum depth of each tree
    seed=42
)

# 3. Train the Model (The "Learning" Phase)
print("Training the model on the 80% Training Set... (This might take a minute)")
gbt_model = gbt.fit(train_data)

# 4. Test the Model (The "Final Exam")
print("Making predictions on the 20% unseen Testing Set...")
predictions = gbt_model.transform(test_data)

# 5. Evaluate Performance
# AUC (Area Under ROC) is the gold standard for binary classification. 
# 0.5 is random guessing, 1.0 is perfect prediction.
evaluator_auc = BinaryClassificationEvaluator(labelCol="churned", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_score = evaluator_auc.evaluate(predictions)

# Accuracy (Percentage of correct predictions)
evaluator_acc = MulticlassClassificationEvaluator(labelCol="churned", predictionCol="prediction", metricName="accuracy")
accuracy_score = evaluator_acc.evaluate(predictions)

print("\n" + "=" * 50)
print("🏆 MODEL EVALUATION METRICS 🏆")
print("=" * 50)
print(f"AUC-ROC Score: {auc_score:.4f} (Closer to 1.0 is better)")
print(f"Accuracy:      {accuracy_score * 100:.2f}%")

# 6. Business Insights: Feature Importance
# Let's prove the model discovered our hidden business rules!
print("\nExtracting Feature Importances...")
feature_cols = [
    "tenure_months", 
    "monthly_premium", 
    "support_calls_last_6m", 
    "payment_delays_days", 
    "claims_filed"
]
importances = gbt_model.featureImportances.toArray()

# Create a DataFrame to display the importances cleanly
importance_df = spark.createDataFrame(
    list(zip(feature_cols, [float(i) for i in importances])), 
    schema=["Feature", "Importance_Weight"]
).orderBy(col("Importance_Weight").desc())

print("✅ Modeling complete! Here is what is driving customer churn:")
display(importance_df)
