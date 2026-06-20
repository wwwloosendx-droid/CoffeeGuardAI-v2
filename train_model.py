import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# ---------------- DATASET PATH ----------------
dataset_path = r"C:\Users\hp\Desktop\Coffee and Cashew Nut Dataset\Coffee\Coffee"

X = []
y = []

# Helper function to extract enhanced features
def extract_features(img):
    """Extract multiple feature types for better classification"""
    features = []
    
    # 1. RGB Color means
    features.extend([
        np.mean(img[:, :, 0]),
        np.mean(img[:, :, 1]),
        np.mean(img[:, :, 2])
    ])
    
    # 2. RGB Color standard deviations
    features.extend([
        np.std(img[:, :, 0]),
        np.std(img[:, :, 1]),
        np.std(img[:, :, 2])
    ])
    
    # 3. RGB Color histograms (16 bins each)
    for i in range(3):
        hist = np.histogram(img[:, :, i], bins=16, range=(0, 256))[0]
        features.extend(hist)
    
    # 4. Convert to HSV and extract statistics
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    features.append(np.mean(hsv[:, :, 0]))  # Hue mean
    features.append(np.mean(hsv[:, :, 1]))  # Saturation mean
    features.append(np.mean(hsv[:, :, 2]))  # Value mean
    features.append(np.std(hsv[:, :, 0]))   # Hue std
    features.append(np.std(hsv[:, :, 1]))   # Saturation std
    features.append(np.std(hsv[:, :, 2]))   # Value std
    
    # 5. Convert to grayscale and compute texture features
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Laplacian edge detection (texture/roughness)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    features.append(np.mean(np.abs(laplacian)))
    features.append(np.std(np.abs(laplacian)))
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.mean(edges))
    features.append(np.sum(edges > 0) / edges.size)
    
    # 6. Sobel edge detection (directional edges)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    features.append(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    features.append(np.std(np.sqrt(sobelx**2 + sobely**2)))
    
    # 7. Contrast and brightness
    features.append(np.max(gray) - np.min(gray))  # Contrast
    features.append(np.mean(gray))  # Brightness
    
    # 8. Color distribution in different regions
    h, w = img.shape[:2]
    for row_split in range(2):
        for col_split in range(2):
            r_start = (h // 2) * row_split
            r_end = (h // 2) * (row_split + 1)
            c_start = (w // 2) * col_split
            c_end = (w // 2) * (col_split + 1)
            region = gray[r_start:r_end, c_start:c_end]
            features.append(np.mean(region))
    
    return features

# ---------------- LOOP THROUGH BATCHES ----------------
for batch in os.listdir(dataset_path):
    batch_path = os.path.join(dataset_path, batch)

    images_path = os.path.join(batch_path, "images")

    if not os.path.exists(images_path):
        print(f"Skipping {batch} (no images folder)")
        continue

    print(f"Loading {batch}")

    # Loop images
    for img_name in os.listdir(images_path):
        img_path = os.path.join(images_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        # resize
        img = cv2.resize(img, (128, 128))

        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # extract enhanced features
        features = extract_features(img)

        X.append(features)

        # Label
        y.append(batch)

# ---------------- CHECK DATA ----------------
print("Total samples collected:", len(X))

if len(X) == 0:
    print("ERROR: No images found. Check dataset structure!")
    exit()

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Train-test split with stratification for balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest with balanced class weights
model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\n--- Training Random Forest Model ---")
model.fit(X_train, y_train)

# Cross-validation for robust accuracy estimate
print("\n--- Cross-Validation Results ---")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Evaluation on test set
print("\n--- Test Set Evaluation ---")
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"Test Accuracy: {acc:.4f}")

# Detailed classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, pred))

# Confusion matrix
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, pred))

# Save model
print("\n--- Saving Model ---")
pickle.dump(model, open("coffee_model.pkl", "wb"))
print("✓ Model saved to 'coffee_model.pkl'")

print("\n✓ MODEL TRAINING COMPLETED SUCCESSFULLY")