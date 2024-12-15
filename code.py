import cv2
import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

# Sampling and statistics setup
frame_counter = 0
sample_interval = 10  # Process every 10th frame
emotion_data = {"angry": [], "disgust": [], "fear": [], "happy": [], "sad": [], "surprise": [], "neutral": []}
dominant_emotions = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video.")
        break

    # Increment frame counter
    frame_counter += 1

    # Convert frame to grayscale for Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face detected
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the face region for emotion analysis
        face_img = frame[y:y+h, x:x+w]

        try:
            # Analyze the face for emotion
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

            # Handle output format based on DeepFace version
            if isinstance(analysis, list):
                emotion_scores = analysis[0]['emotion']
                dominant_emotion = analysis[0]['dominant_emotion']
            else:
                emotion_scores = analysis['emotion']
                dominant_emotion = analysis['dominant_emotion']

            # Record probabilities for each emotion
            for emotion, score in emotion_scores.items():
                emotion_data[emotion.lower()].append(score)

            # Record the dominant emotion for this frame
            dominant_emotions.append(dominant_emotion)

            # Display the most dominant emotion above the face rectangle
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        except Exception as e:
            print(f"Error in emotion analysis: {e}")

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break loop on space bar press
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Analyze Across Frames
emotion_means = {emotion: np.mean(scores) if scores else 0 for emotion, scores in emotion_data.items()}
emotion_variances = {emotion: np.var(scores) if scores else 0 for emotion, scores in emotion_data.items()}
dominant_emotion_counts = {emotion: dominant_emotions.count(emotion) for emotion in set(dominant_emotions)}

# Mental Well-Being Analysis
def analyze_wellbeing(dominant_emotions):
    transitions = sum(1 for i in range(1, len(dominant_emotions)) if dominant_emotions[i] != dominant_emotions[i-1])
    predominant_negative = sum(dominant_emotions.count(emotion) for emotion in ['angry', 'sad', 'fear', 'disgust'])
    predominant_positive = dominant_emotions.count('happy')
    neutral_count = dominant_emotions.count('neutral')

    # Emotional Stability Analysis
    stability = "Stable" if transitions / len(dominant_emotions) < 0.3 else "Unstable"

    # Risk Analysis for Depression, Anxiety, or Stress
    depression_risk = "High" if dominant_emotions.count('sad') / len(dominant_emotions) > 0.3 else "Low"
    anxiety_risk = "High" if dominant_emotions.count('fear') / len(dominant_emotions) > 0.2 else "Low"
    stress_risk = "High" if transitions / len(dominant_emotions) > 0.4 or predominant_negative > predominant_positive else "Low"

    # Positive and Neutral Proportions
    positive_dominance = "High" if predominant_positive > predominant_negative else "Low"
    neutral_proportion = "High" if neutral_count / len(dominant_emotions) > 0.4 else "Moderate"

    wellbeing_report = {
        "Emotional Stability": stability,
        "Risk of Depression": depression_risk,
        "Risk of Anxiety": anxiety_risk,
        "Risk of Stress": stress_risk,
        "Positive Dominance": positive_dominance,
        "Neutral Proportion": neutral_proportion,
    }
    return wellbeing_report

wellbeing_report = analyze_wellbeing(dominant_emotions)

# Display Results
print("Emotion Analysis:")
print("Mean Scores:", emotion_means)
print("Variance Scores:", emotion_variances)
print("Dominant Emotion Counts:", dominant_emotion_counts)
print("\nMental Well-Being Report:")
for key, value in wellbeing_report.items():
    print(f"{key}: {value}")

# Plot Emotion Statistics
plt.figure(figsize=(14, 8))

# Bar plot for mean probabilities
plt.subplot(2, 2, 1)
plt.bar(emotion_means.keys(), emotion_means.values(), color='skyblue')
plt.title('Mean Emotion Probabilities')
plt.ylabel('Probability')
plt.xlabel('Emotions')

# Pie chart for dominant emotion counts
plt.subplot(2, 2, 2)
plt.pie(dominant_emotion_counts.values(), labels=dominant_emotion_counts.keys(), autopct='%1.1f%%', colors=plt.cm.Paired.colors)
plt.title('Dominant Emotion Frequency')

# Text-based Well-Being Report
plt.subplot(2, 1, 2)
plt.axis('off')
plt.text(0.5, 0.5, "\n".join([f"{key}: {value}" for key, value in wellbeing_report.items()]),
         fontsize=12, ha='center', va='center', wrap=True, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
plt.title('Mental Well-Being Analysis')

plt.tight_layout()
plt.show()
