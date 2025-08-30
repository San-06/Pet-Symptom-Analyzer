import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Pet Symptom Analyzer 🐾")
st.write("Upload a text file or enter symptoms to get predicted condition, structured insights, and download the result.")

# --- Function to preprocess and predict ---
def predict_condition(text_list):
    vectors = vectorizer.transform(text_list)
    predictions = model.predict(vectors)
    return predictions

# --- Function to extract insights ---
def generate_insights(df):
    insights = df['Predicted Condition'].value_counts().reset_index()
    insights.columns = ['Condition', 'Count']
    return insights

# --- Input Options ---
input_mode = st.radio("Choose Input Type:", ("Enter Text", "Upload File"))

if input_mode == "Enter Text":
    user_text = st.text_area("Enter pet symptoms here:")
    if st.button("Predict"):
        if user_text.strip():
            pred = predict_condition([user_text])[0]
            st.success(f"Predicted Condition: **{pred}**")
            result_df = pd.DataFrame([{"Input Text": user_text, "Predicted Condition": pred}])
            st.dataframe(result_df)

            st.subheader("📊 Actionable Insights")
            st.write("Since it's only one entry, no chart is generated.")
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name="prediction_result.csv", mime="text/csv")
        else:
            st.warning("Please enter some text.")

else:
    uploaded_file = st.file_uploader("Upload a .txt file with pet symptom descriptions", type=["txt"])
    if uploaded_file is not None:
        file_content = uploaded_file.read().decode("utf-8")
        text_lines = [line.strip() for line in file_content.strip().split('\n') if line.strip()]

        if len(text_lines) == 0:
            st.warning("The uploaded file is empty or invalid.")
        else:
            predictions = predict_condition(text_lines)
            result_df = pd.DataFrame({"Input Text": text_lines, "Predicted Condition": predictions})
            st.subheader("🗂️ Structured Table")
            st.dataframe(result_df)

            st.subheader("📊 Actionable Insights")
            insight_df = generate_insights(result_df)
            st.dataframe(insight_df)

            # Plot
            fig, ax = plt.subplots()
            ax.bar(insight_df['Condition'], insight_df['Count'], color='skyblue')
            plt.xticks(rotation=45)
            plt.title("Condition Frequency")
            st.pyplot(fig)

            # Download
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Result CSV", data=csv, file_name="predicted_conditions.csv", mime="text/csv")

