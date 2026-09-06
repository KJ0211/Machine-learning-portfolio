Machine Learning Portfolio

Applied ML projects spanning NLP (transformers, text classification, knowledge tracing) and sensor-data analysis (haptics, human activity recognition), built through MSc Human-Computer Interaction coursework/dissertation work and independent projects.

NLP & Education ML

Automated Essay Scoring with a Fine-Tuned Transformer Fine-tuned DistilBERT to score student essays (Kaggle ASAP-AES). Beat a word-count baseline (0.78 QWK) with 0.86 QWK, while identifying essay length as a known confound in this dataset.

CEFR-Level Text Classifier Classified English text by proficiency level (A1–C2) with a fine-tuned transformer, improving macro-F1 from 0.59 (TF-IDF baseline) to 0.65. Errors cluster almost entirely on adjacent CEFR levels — evidence the model learned the level ordering, not just surface features.

Knowledge Tracing on Duolingo Learner Data Predicted per-word answer correctness from 2.6M Duolingo learner interactions, using leakage-safe historical-performance features and gradient boosting. AUC 0.73 vs. 0.66 for a naive history-only baseline; recovers 53% of learner errors at a tuned decision threshold.

Sensor Data & Human Activity

Machine Learning for Characterising Caring Touch in Haptic and Robotic Systems MSc dissertation (in progress): a reproducible pipeline for pressure-sensor data from a sensorised infant doll, extracting interpretable touch-pattern features (intensity, rhythm, smoothness) across caregiving tasks.

Classifying Human Interaction Patterns in Inertial Sensor Data Random Forest classifier on the UCI Human Activity Recognition dataset — 93% test accuracy distinguishing six activities from smartphone accelerometer/gyroscope data.

Investigative Data Preparation with Pandas An end-to-end data-cleaning and feature-engineering pipeline over transaction and suspect records — joins, outlier detection, aggregation, and export of analysis-ready datasets.

Tools

Python, Pandas, NumPy, Scikit-learn, PyTorch, Hugging Face Transformers, Matplotlib/Seaborn, Jupyter.
