# CoffeeGuard AI — Revised Research Objectives

## 1.3 Objectives of the Study

### 1.3.1 General Objective
To design, develop, and evaluate a web-based machine learning system, CoffeeGuard AI, that detects coffee leaf diseases from leaf images and uses the detected disease severity to estimate the likely impact on coffee yield, in order to support timely, evidence-based decision-making among smallholder coffee farmers in Uganda.

### 1.3.2 Specific Objectives

1. **To collect and preprocess a coffee leaf image dataset** representative of healthy leaves and common Ugandan coffee diseases (notably coffee leaf rust and coffee wilt disease), including image cleaning, augmentation, and labelling, in order to produce a dataset suitable for supervised deep learning.

2. **To design and train a convolutional neural network (CNN) model**, using transfer learning on a pre-trained architecture (ResNet-50), for the automatic detection and classification of coffee leaf diseases by severity category.

3. **To develop a rule-based yield-impact estimation model** that translates the CNN's predicted disease class and confidence score into an estimated yield outlook and an actionable farmer recommendation, reflecting the relationship between disease severity and expected crop loss reported in agronomic literature.

4. **To design and implement a web-based decision-support application** (CoffeeGuard AI) that allows farmers to register, upload leaf images, view diagnosis results with visual model-explainability (Grad-CAM heatmaps), and track their scan history and yield-outlook trends over time on a dashboard.

5. **To evaluate the performance of the developed system** using standard classification metrics (accuracy, precision, recall, F1-score, and confusion matrix analysis) for the disease-detection model, and through usability feedback for the web application, in order to assess its reliability and practical value for end users.

---

## Notes on the Revisions Made

- The **general objective** was broadened slightly to explicitly name the deliverable (a web-based system) and its purpose (decision support for farmers), which gives examiners a clearer picture of scope than a purely technical statement.
- **Objective 3** was reframed from "developing a predictive model that estimates yield" to a **rule-based / heuristic yield-impact estimation** objective. This is an honest description of what the current implementation does (it maps disease class + confidence to a yield percentage using domain-informed thresholds) and avoids overclaiming a statistically trained regression yield model that does not yet exist. If you later train an actual regression model on real yield data (e.g., from farm records), you can upgrade this objective to: *"To train and validate a regression-based yield prediction model using historical yield and disease-severity data."*
- **Objective 4** was added because your dissertation should be assessed not only on the CNN but also on the engineering of a usable system — this is where your dashboard, authentication, and database work earns marks.
- **Objective 5** keeps evaluation as its own objective (as you had it) but ties it to concrete, defensible metrics examiners expect to see in your results chapter, plus usability evaluation for the application layer.

This structure gives you five objectives that map cleanly onto five result/discussion sections: (1) Data, (2) Model architecture & training, (3) Yield-impact logic, (4) System design & implementation, (5) Evaluation. That mapping will make your dissertation much easier to write and defend.
