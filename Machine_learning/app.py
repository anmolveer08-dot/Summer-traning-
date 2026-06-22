import pickle

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

import streamlit as st

# ================= PAGE SETTINGS =================

st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ================= SIDEBAR =================

st.sidebar.title("📊 Spam Detection Dashboard")

st.sidebar.metric(
    label="Model Accuracy",
    value="95%"
)

st.sidebar.metric(
    label="Algorithm",
    value="Logistic Regression"
)

st.sidebar.metric(
    label="Vectorizer",
    value="TF-IDF"
)

st.sidebar.success("✅ Model Ready")

# ================= CUSTOM CSS =================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #312e81
    );
}

h1 {
    color: white;
    text-align: center;
}

p {
    color: white;
}

textarea {
    border-radius: 15px !important;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================

st.title("📧 AI Email Spam Detection System")

st.write(
    "Paste an email or message below and click Analyze Message."
)

# ================= INPUT BOX =================

message = st.text_area(
    "Paste Email or Message",
    height=250
)

# ================= PREDICTION =================

if st.button("🚀 Analyze Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        msg_vector = vectorizer.transform([message])

        prediction = model.predict(msg_vector)
        probability = model.predict_proba(msg_vector)

        ham_score = probability[0][0] * 100
        spam_score = probability[0][1] * 100

        if prediction[0] == 1:

            st.error(
                f"⚠ SPAM DETECTED ({spam_score:.2f}%)"
            )

            st.progress(int(spam_score))

        else:

            st.success(
                f"✅ SAFE MESSAGE ({ham_score:.2f}%)"
            )

            st.progress(int(ham_score))

        st.subheader("Confidence Scores")

        st.write(
            f"Spam Probability: {spam_score:.2f}%"
        )

        st.write(
            f"Ham Probability: {ham_score:.2f}%"
        )




